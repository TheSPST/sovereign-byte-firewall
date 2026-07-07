import os
import csv
import json
import math
import hashlib
import datetime
from collections import Counter
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
    
    Data Provenance: Automatically hashes dataset and caches metadata manifests.
    Ground Truth Helper: Extracts and writes known anomalies (SYN floods & entropy spikes) 
    to a buffered CSV file for auditing.
    """
    def __init__(self, pcap_path, max_sequence_length=8192):
        super().__init__()
        self.pcap_path = pcap_path
        self.max_sequence_length = max_sequence_length
        self.file_hash = None
        self.cached_sequences = None
        
        # Core checks
        if not os.path.exists(pcap_path):
            raise FileNotFoundError(f"Target PCAP file not found at: {pcap_path}")
            
        # 1. Load manifest cache or compute hash streaming-wise
        self._load_or_compute_hash()

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
        counts = Counter(packet_data)
        total_len = len(packet_data)
        entropy = 0.0
        for count in counts.values():
            p = count / total_len
            entropy -= p * math.log2(p)
        return entropy

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

        try:
            with RawPcapReader(self.pcap_path) as pcap_reader:
                current_sequence = []
                packet_index = 0
                sequence_count = 0
                byte_offset = 0
                
                for packet_data, _metadata in pcap_reader:
                    # Increment global byte offset
                    packet_len = len(packet_data)
                    
                    # 1. Ground Truth Anomaly Labeling & Analysis
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

                    # 2. Yield sequence slicing for dataloader
                    if packet_index % num_workers == worker_id:
                        for byte in packet_data:
                            current_sequence.append(byte)
                            if len(current_sequence) == self.max_sequence_length:
                                yield torch.tensor(current_sequence, dtype=torch.long)
                                sequence_count += 1
                                current_sequence = []
                                
                    packet_index += 1
                    byte_offset += packet_len
                
                # Yield any remaining trailing bytes padded with 0x00
                if current_sequence:
                    remainder = self.max_sequence_length - len(current_sequence)
                    current_sequence.extend([0] * remainder)
                    yield torch.tensor(current_sequence, dtype=torch.long)
                    sequence_count += 1
            
            # Atomic Manifest update when loading finishes (only write for main process / worker 0)
            if worker_id == 0:
                self._write_manifest_atomic(sequence_count)
                
        finally:
            csv_file.close()

def get_pcap_dataloader(pcap_path, batch_size=32, num_workers=0, max_sequence_length=8192):
    """
    Factory function to create a PyTorch DataLoader for the PCAP streaming dataset.
    """
    dataset = RawPcapIterableDataset(pcap_path, max_sequence_length=max_sequence_length)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
