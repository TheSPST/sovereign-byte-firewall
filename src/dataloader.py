import os
import csv
import json
import math
import hashlib
import datetime
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether
from scapy.layers.inet import TCP

class RawPcapIterableDataset(IterableDataset):
    """
    A PyTorch IterableDataset that streams raw bytes from a PCAP file packet-by-packet.
    It yields sequences of a fixed length (max_sequence_length), zero-padded at the end.
    Supports multi-process loading by partitioning packets among workers.
    
    Optimized:
      1. Zero-Copy Tensor Striding via torch.as_strided() (O(1) memory view layouts).
      2. Vectorized Shannon Entropy via torch.bincount (no Python loop / Counter).
      3. Strict Trailing Remainder bounds calculation to prevent out-of-bounds.
      4. Tensor cloning on yield to prevent OOM memory pointer leaks of large file chunks.
    """
    def __init__(self, pcap_path, max_sequence_length=8192, stride=None, mask_addresses=True):
        super().__init__()
        self.pcap_path = pcap_path
        self.max_sequence_length = max_sequence_length
        # Default stride to max_sequence_length (non-overlapping) if not specified
        self.stride = stride if stride is not None else max_sequence_length
        self.mask_addresses = mask_addresses
        self.file_hash = None
        self.cached_sequences = None
        
        # Core checks
        if not os.path.exists(pcap_path):
            raise FileNotFoundError(f"Target PCAP file not found at: {pcap_path}")
            
        # Load manifest cache or compute hash streaming-wise
        self._load_or_compute_hash()

    def __len__(self):
        if self.cached_sequences is not None:
            return self.cached_sequences
        # Fallback estimate based on file size and stride
        try:
            file_size = os.path.getsize(self.pcap_path)
            # Assuming average packet payload content is ~85% of PCAP size
            estimated_bytes = int(file_size * 0.85)
            estimated_sequences = max(1, estimated_bytes // self.stride)
            return estimated_sequences
        except Exception:
            return 1000  # Safe default fallback

    def _load_or_compute_hash(self):
        filename = os.path.basename(self.pcap_path)
        manifest_dir = "./data/manifests"
        os.makedirs(manifest_dir, exist_ok=True)
        manifest_path = os.path.join(manifest_dir, f"{filename}.json")
        
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                
                # Check modification time
                pcap_mtime = os.path.getmtime(self.pcap_path)
                cached_mtime = manifest.get("file_mtime")
                
                # Trust cache if PCAP file has not been modified
                if cached_mtime == pcap_mtime:
                    self.file_hash = manifest.get("file_hash")
                    self.cached_sequences = manifest.get("num_sequences_extracted")
                    print(f"Cache Hit: Loaded SHA-256 from manifest for '{filename}'")
                    return
            except Exception as e:
                print(f"Warning: Failed to read cached manifest: {e}. Recomputing hash...")
                
        # Cache Miss: Compute streaming hash
        print(f"Cache Miss: Computing streaming SHA-256 hash for '{filename}'...")
        self.file_hash = self._compute_streaming_sha256()
        print(f"Hash secured: {self.file_hash}")

    def _compute_streaming_sha256(self):
        sha256_hash = hashlib.sha256()
        with open(self.pcap_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _write_manifest_atomic(self, num_sequences):
        filename = os.path.basename(self.pcap_path)
        manifest_dir = "./data/manifests"
        os.makedirs(manifest_dir, exist_ok=True)
        manifest_path = os.path.join(manifest_dir, f"{filename}.json")
        
        data = {
            "source_filename": filename,
            "file_hash": self.file_hash,
            "file_mtime": os.path.getmtime(self.pcap_path),
            "timestamp_processed": datetime.datetime.now().isoformat(),
            "num_sequences_extracted": num_sequences
        }
        
        temp_path = manifest_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        os.replace(temp_path, manifest_path)
        self.cached_sequences = num_sequences
        print(f"Atomic manifest updated at: {manifest_path}")

    def _calculate_packet_entropy(self, packet_data):
        if not packet_data:
            return 0.0
        # Vectorized Shannon Entropy via torch.bincount (no Python loop / Counter)
        pkt_tensor = torch.tensor(list(packet_data), dtype=torch.long)
        counts = torch.bincount(pkt_tensor, minlength=256).float()
        probs = counts / len(packet_data)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-9))
        return entropy.item()

    def _mask_packet_addresses(self, packet_data):
        """
        Dynamically masks IP, MAC, TCP Options, and Encrypted TLS Application Data
        in raw packet bytes using a dynamic protocol parser. Supports VLAN, QinQ,
        MPLS, IPv4, IPv6, ARP, and TCP/TLS.
        """
        pkt_bytes = bytearray(packet_data)
        n = len(pkt_bytes)
        
        # 1. Layer 2 Blanking (Static MACs)
        if n >= 12:
            pkt_bytes[0:12] = b'\x00' * 12
            
        # 2. Dynamic Layer 3 Offset Calculation (supporting VLAN, QinQ, MPLS)
        if n < 14:
            return bytes(pkt_bytes)
            
        offset = 12
        ethertype = (pkt_bytes[offset] << 8) | pkt_bytes[offset + 1]
        offset += 2
        
        # Loop to handle stacked VLAN tags (QinQ: 0x8100, 0x88a8, 0x9100, 0x9200)
        while ethertype in (0x8100, 0x88a8, 0x9100, 0x9200):
            if n < offset + 4:
                return bytes(pkt_bytes)
            ethertype = (pkt_bytes[offset + 2] << 8) | pkt_bytes[offset + 3]
            offset += 4
            
        l3_offset = offset
        resolved_ethertype = ethertype
        
        # Handle MPLS Unicast/Multicast (0x8847 / 0x8848)
        if resolved_ethertype in (0x8847, 0x8848):
            while True:
                if n < l3_offset + 4:
                    return bytes(pkt_bytes)
                bos = pkt_bytes[l3_offset + 2] & 0x01
                l3_offset += 4
                if bos:
                    break
            if n < l3_offset + 1:
                return bytes(pkt_bytes)
            version = (pkt_bytes[l3_offset] >> 4) & 0x0F
            if version == 4:
                resolved_ethertype = 0x0800
            elif version == 6:
                resolved_ethertype = 0x86DD
            else:
                resolved_ethertype = None
                
        # 3. Protocol-Specific L3 Address Masking
        l4_offset = None
        is_tcp = False
        
        if resolved_ethertype == 0x0800:  # IPv4
            if n >= l3_offset + 20:
                pkt_bytes[l3_offset + 12 : l3_offset + 20] = b'\x00' * 8
                
                protocol = pkt_bytes[l3_offset + 9]
                if protocol == 6:  # TCP
                    ihl = pkt_bytes[l3_offset] & 0x0F
                    l4_offset = l3_offset + (ihl * 4)
                    is_tcp = True
                    
        elif resolved_ethertype == 0x86DD:  # IPv6
            if n >= l3_offset + 40:
                pkt_bytes[l3_offset + 8 : l3_offset + 40] = b'\x00' * 32
                
                next_header = pkt_bytes[l3_offset + 6]
                if next_header == 6:  # TCP
                    l4_offset = l3_offset + 40
                    is_tcp = True
                    
        elif resolved_ethertype == 0x0806:  # ARP
            if n >= l3_offset + 28:
                pkt_bytes[l3_offset + 8 : l3_offset + 28] = b'\x00' * 20
                
        # 4. Layer 4 (TCP) & Layer 7 (TLS) Masking
        if is_tcp and l4_offset is not None:
            if n >= l4_offset + 20:
                tcp_data_offset = (pkt_bytes[l4_offset + 12] >> 4) & 0x0F
                tcp_header_len = tcp_data_offset * 4
                
                # Mask variable TCP options (if TCP options are present, i.e., header len > 20)
                if tcp_header_len > 20 and n >= l4_offset + tcp_header_len:
                    pkt_bytes[l4_offset + 20 : l4_offset + tcp_header_len] = b'\x00' * (tcp_header_len - 20)
                    
                # Mask Encrypted TLS Payloads
                sport = (pkt_bytes[l4_offset] << 8) | pkt_bytes[l4_offset + 1]
                dport = (pkt_bytes[l4_offset + 2] << 8) | pkt_bytes[l4_offset + 3]
                if sport == 443 or dport == 443:
                    tls_offset = l4_offset + tcp_header_len
                    if n >= tls_offset + 5:
                        tls_content_type = pkt_bytes[tls_offset]
                        # 0x17 represents Application Data (Encrypted Payload)
                        if tls_content_type == 0x17:
                            pkt_bytes[tls_offset + 5 : n] = b'\x00' * (n - (tls_offset + 5))
                            
        return bytes(pkt_bytes)

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        # Prepare buffered anomaly CSV writer
        anomaly_csv_path = "./data/anomaly_labels.csv"
        os.makedirs(os.path.dirname(anomaly_csv_path), exist_ok=True)
        file_exists = os.path.exists(anomaly_csv_path)
        
        # Open file in append mode with buffering=1 (line-buffered)
        csv_file = open(anomaly_csv_path, "a", newline="", buffering=1, encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        if not file_exists:
            csv_writer.writerow(["pcap_file", "packet_index", "byte_offset", "anomaly_type", "entropy_value"])

        # Configure sliding window and stride dimensions
        W = self.max_sequence_length
        S = self.stride
        
        # 10MB streaming chunks buffer limit
        buffer_limit = 10 * 1024 * 1024 

        try:
            with RawPcapReader(self.pcap_path) as pcap_reader:
                buffer = []
                packet_index = 0
                sequence_count = 0
                byte_offset = 0
                
                for packet_data, _metadata in pcap_reader:
                    packet_len = len(packet_data)
                    
                    # 1. Vectorized Anomaly Labeling & Analysis
                    is_syn = False
                    try:
                        pkt = Ether(packet_data)
                        if pkt.haslayer(TCP):
                            flags = int(pkt[TCP].flags)
                            # Check for TCP SYN flags (SYN=2 set, ACK=16 is not set)
                            if (flags & 0x02) and not (flags & 0x10):
                                is_syn = True
                    except Exception:
                        pass
                    
                    entropy = self._calculate_packet_entropy(packet_data)
                    
                    # Label anomalies in append buffer
                    if is_syn or entropy < 1.0 or (entropy > 7.7 and packet_len > 100):
                        anomaly_type = "TCP_SYN_Flood" if is_syn else "Abnormal_Entropy"
                        csv_writer.writerow([
                            os.path.basename(self.pcap_path), 
                            packet_index, 
                            byte_offset, 
                            anomaly_type, 
                            f"{entropy:.4f}"
                        ])

                    # 2. Accumulate packet bytes for partitioned worker
                    if packet_index % num_workers == worker_id:
                        data_to_append = self._mask_packet_addresses(packet_data) if self.mask_addresses else packet_data
                        buffer.extend(data_to_append)
                        
                        # Process buffer if it exceeds the limit
                        if len(buffer) >= buffer_limit:
                            flat_bytes = torch.tensor(buffer, dtype=torch.long)
                            N = len(flat_bytes)
                            
                            # Strict Trailing Remainder bounds calculation to prevent out-of-bounds
                            num_windows = max(0, (N - W) // S + 1)
                            if num_windows > 0:
                                limit_bytes = num_windows * S + (W - S)
                                windows = torch.as_strided(flat_bytes[:limit_bytes], size=(num_windows, W), stride=(S, 1))
                                for window in windows:
                                    # Tensor cloning on yield to prevent OOM memory pointer leaks
                                    yield window.clone()
                                    sequence_count += 1
                                # Keep remainder
                                buffer = buffer[num_windows * S:]
                                
                    packet_index += 1
                    byte_offset += packet_len
                
                # Process remaining buffer at the end of the file
                if len(buffer) > 0:
                    flat_bytes = torch.tensor(buffer, dtype=torch.long)
                    N = len(flat_bytes)
                    num_windows = max(0, (N - W) // S + 1)
                    
                    if num_windows > 0:
                        limit_bytes = num_windows * S + (W - S)
                        windows = torch.as_strided(flat_bytes[:limit_bytes], size=(num_windows, W), stride=(S, 1))
                        for window in windows:
                            yield window.clone()
                            sequence_count += 1
                        remainder_start = num_windows * S
                    else:
                        remainder_start = 0
                        
                    # Yield trailing bytes padded with 0
                    trailing = flat_bytes[remainder_start:]
                    if len(trailing) > 0:
                        pad_len = W - len(trailing)
                        padded_tensor = torch.cat([trailing, torch.zeros(pad_len, dtype=torch.long)])
                        yield padded_tensor.clone()
                        sequence_count += 1
            
            # Atomic Manifest update when loading finishes (only write for main process / worker 0)
            if worker_id == 0:
                self._write_manifest_atomic(sequence_count)
                
        finally:
            csv_file.close()

def get_pcap_dataloader(pcap_path, batch_size=32, num_workers=0, max_sequence_length=8192, stride=None, mask_addresses=True):
    """
    Factory function to create a PyTorch DataLoader for the PCAP streaming dataset.
    """
    dataset = RawPcapIterableDataset(
        pcap_path, 
        max_sequence_length=max_sequence_length, 
        stride=stride, 
        mask_addresses=mask_addresses
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
