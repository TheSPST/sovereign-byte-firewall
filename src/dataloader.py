import os
import csv
import json
import math
import time
import random
import hashlib
import datetime
from collections import defaultdict, deque
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, TCP

# TCP ports treated as implicit-TLS: record-level masking of encrypted payload.
# 443 HTTPS, 8443 HTTPS-alt, 993 IMAPS, 995 POP3S, 465 SMTPS, 990 FTPS, 636 LDAPS
TLS_PORTS = frozenset({443, 8443, 993, 995, 465, 990, 636})
# TCP ports treated as SSH: mask all payload except the plaintext version banner.
SSH_PORTS = frozenset({22})
# UDP ports whose payload is (essentially) always encrypted:
# 443 QUIC/HTTP3, 4500 IPsec NAT-T, 51820 WireGuard, 1194 OpenVPN
UDP_ENCRYPTED_PORTS = frozenset({443, 4500, 51820, 1194})

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
      5. Bounded shuffle buffer: since packets are streamed in strict file (chronological)
         order, training on them unshuffled means gradient updates within any stretch of
         steps are drawn from whatever narrow slice of the day the reader currently sits
         in — a highly non-stationary signal. Windows are pooled into a fixed-size buffer,
         shuffled, and emitted in randomized order to break that correlation while keeping
         memory bounded (this is the standard streaming shuffle-buffer technique).
    """
    def __init__(self, pcap_path, max_sequence_length=8192, stride=None, mask_addresses=True, shuffle_buffer_windows=4096, label_anomalies=True):
        super().__init__()
        self.pcap_path = pcap_path
        self.max_sequence_length = max_sequence_length
        # Default stride to max_sequence_length (non-overlapping) if not specified
        self.stride = stride if stride is not None else max_sequence_length
        self.mask_addresses = mask_addresses
        self.shuffle_buffer_windows = shuffle_buffer_windows
        # When False, skips the per-packet scapy parse + entropy calc + CSV write
        # entirely. The anomaly CSV is a side-channel feature, NOT a training input
        # (the loss is unsupervised next-byte prediction) — disabling it roughly
        # halves per-packet CPU cost, which matters when the dataloader is the
        # bottleneck feeding an A100. Evaluation harnesses should always pass False
        # so scoring runs don't pollute data/anomaly_labels.csv with junk rows.
        self.label_anomalies = label_anomalies
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
            if self.pcap_path.endswith(".gz"):
                # PCAPs typically compress to ~20% of their original size, so scale by 5.0
                file_size = int(file_size * 5.0)
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

    def _mask_packet_addresses(self, packet_data, stream_tls_state=None):
        """
        Dynamically masks IP, MAC, TCP Options, and Encrypted TLS Application Data
        in raw packet bytes using a dynamic protocol parser. Supports VLAN, QinQ,
        MPLS, IPv4, IPv6, ARP, and TCP/TLS.

        stream_tls_state: optional dict (shared across all packets in this
        iterator's __iter__ call), keyed by (src_ip, dst_ip, sport, dport).
        Value: (remaining_bytes_of_in_progress_record, mask_flag). Entries
        persist with remaining=0 once a stream is confirmed TLS — used to
        decide whether unparseable payload on a TLS port is mid-stream
        ciphertext we lost track of (mask it) vs. genuinely non-TLS traffic
        (leave it visible to the model).

        FIX history:
        (a) A single TLS record is very often larger than one TCP segment
            (records up to 16KB vs. ~1460-byte MSS), so most HTTPS transfers
            span many packets. The original code only checked the FIRST byte
            of each packet's payload for the 0x17 Application Data marker —
            true only for the packet that starts a record. Continuation
            packets start mid-ciphertext and were passing through UNMASKED
            as raw high-entropy noise. Fixed with a per-stream
            remaining-byte counter.
        (b) Only the first record header per packet was parsed: if record A
            ended mid-packet and record B started in the same packet and
            spilled into the next, B's continuation went unmasked. Fixed by
            walking ALL records in the packet in a loop.
        (c) Cross-packet state now tracks non-0x17 records too (handshake
            certificate chains regularly span packets); their continuation
            bytes are consumed but NOT masked, preventing a desync where a
            handshake continuation byte happens to equal 0x17.
        (d) Port coverage widened from {443} to common implicit-TLS ports,
            plus SSH (port 22): all SSH payload except the plaintext
            "SSH-" version banner is masked, since post-banner SSH traffic
            is (mostly) encrypted and was training as ciphertext noise.
        (e) Stochastic HEADER fields masked. Guiding principle: mask any
            byte that is irreducibly random under benign traffic, keep any
            byte that carries protocol grammar. TCP seq/ack (random ISNs, 8
            bytes/packet — benign traffic is dominated by small ACKs, so
            this was ~15% noise per ACK frame), TCP+IP checksums (derived
            from already-masked fields => effectively random), IPv4 ID,
            IPv6 flow label. Ports, flags, window, TTL, lengths stay
            visible — they carry signal (scans, floods, weird services).
        (f) UDP finally parsed: UDP checksum masked, and payload on
            always-encrypted UDP ports (QUIC 443, WireGuard, OpenVPN,
            IPsec NAT-T) fully masked — QUIC was previously 100% unmasked
            ciphertext in training data.
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
                
        # 3. Protocol-Specific L3 Address & Stochastic-Field Masking
        l4_offset = None
        is_tcp = False
        is_udp = False
        # Raw (pre-mask) address bytes, captured for the TLS stream-state key
        # before the address-blanking below zeroes them out in pkt_bytes.
        raw_src_ip = None
        raw_dst_ip = None

        if resolved_ethertype == 0x0800:  # IPv4
            if n >= l3_offset + 20:
                raw_src_ip = bytes(pkt_bytes[l3_offset + 12 : l3_offset + 16])
                raw_dst_ip = bytes(pkt_bytes[l3_offset + 16 : l3_offset + 20])
                pkt_bytes[l3_offset + 12 : l3_offset + 20] = b'\x00' * 8
                # Stochastic per-packet fields (see docstring FIX (e)):
                pkt_bytes[l3_offset + 4 : l3_offset + 6] = b'\x00' * 2    # IP Identification (random per packet)
                pkt_bytes[l3_offset + 10 : l3_offset + 12] = b'\x00' * 2  # header checksum (derives from masked fields)

                protocol = pkt_bytes[l3_offset + 9]
                ihl = pkt_bytes[l3_offset] & 0x0F
                if protocol == 6:  # TCP
                    l4_offset = l3_offset + (ihl * 4)
                    is_tcp = True
                elif protocol == 17:  # UDP
                    l4_offset = l3_offset + (ihl * 4)
                    is_udp = True

        elif resolved_ethertype == 0x86DD:  # IPv6
            if n >= l3_offset + 40:
                raw_src_ip = bytes(pkt_bytes[l3_offset + 8 : l3_offset + 24])
                raw_dst_ip = bytes(pkt_bytes[l3_offset + 24 : l3_offset + 40])
                pkt_bytes[l3_offset + 8 : l3_offset + 40] = b'\x00' * 32
                # Flow label (20 bits): pseudo-random per flow — mask it.
                pkt_bytes[l3_offset + 1] &= 0xF0
                pkt_bytes[l3_offset + 2 : l3_offset + 4] = b'\x00' * 2

                next_header = pkt_bytes[l3_offset + 6]
                if next_header == 6:  # TCP
                    l4_offset = l3_offset + 40
                    is_tcp = True
                elif next_header == 17:  # UDP
                    l4_offset = l3_offset + 40
                    is_udp = True
                    
        elif resolved_ethertype == 0x0806:  # ARP
            if n >= l3_offset + 28:
                pkt_bytes[l3_offset + 8 : l3_offset + 28] = b'\x00' * 20
                
        # 4. Layer 4 (TCP/UDP) & Layer 7 (TLS/SSH/QUIC) Masking
        if is_udp and l4_offset is not None and n >= l4_offset + 8:
            udp_sport = (pkt_bytes[l4_offset] << 8) | pkt_bytes[l4_offset + 1]
            udp_dport = (pkt_bytes[l4_offset + 2] << 8) | pkt_bytes[l4_offset + 3]
            # UDP checksum: derives from masked fields => effectively random.
            pkt_bytes[l4_offset + 6 : l4_offset + 8] = b'\x00' * 2
            # Always-encrypted UDP protocols (QUIC/VPNs): mask whole payload.
            if udp_sport in UDP_ENCRYPTED_PORTS or udp_dport in UDP_ENCRYPTED_PORTS:
                if n > l4_offset + 8:
                    pkt_bytes[l4_offset + 8 : n] = b'\x00' * (n - (l4_offset + 8))

        if is_tcp and l4_offset is not None:
            if n >= l4_offset + 20:
                tcp_data_offset = (pkt_bytes[l4_offset + 12] >> 4) & 0x0F
                tcp_header_len = tcp_data_offset * 4

                # Stochastic per-packet fields (see docstring FIX (e)):
                pkt_bytes[l4_offset + 4 : l4_offset + 12] = b'\x00' * 8    # seq + ack numbers (random ISNs)
                pkt_bytes[l4_offset + 16 : l4_offset + 18] = b'\x00' * 2   # TCP checksum
                # Kept visible on purpose: ports, flags, window size, urgent
                # pointer — all carry attack-relevant grammar (scans, floods,
                # zero-window stalls, weird service targeting).

                # Mask variable TCP options (if TCP options are present, i.e., header len > 20)
                if tcp_header_len > 20 and n >= l4_offset + tcp_header_len:
                    pkt_bytes[l4_offset + 20 : l4_offset + tcp_header_len] = b'\x00' * (tcp_header_len - 20)

                # Mask Encrypted TLS / SSH Payloads
                sport = (pkt_bytes[l4_offset] << 8) | pkt_bytes[l4_offset + 1]
                dport = (pkt_bytes[l4_offset + 2] << 8) | pkt_bytes[l4_offset + 3]
                payload_start = l4_offset + tcp_header_len
                payload_end = n

                if sport in TLS_PORTS or dport in TLS_PORTS:
                    stream_key = (raw_src_ip, raw_dst_ip, sport, dport)
                    pos = payload_start

                    # --- 1. Consume continuation of a record from an earlier packet ---
                    confirmed_tls_stream = False
                    if stream_tls_state is not None and stream_key in stream_tls_state:
                        confirmed_tls_stream = True
                        remaining, mask_flag = stream_tls_state[stream_key]
                        if remaining > 0 and pos < payload_end:
                            consumed = min(remaining, payload_end - pos)
                            if mask_flag:
                                pkt_bytes[pos: pos + consumed] = b'\x00' * consumed
                            pos += consumed
                            stream_tls_state[stream_key] = (remaining - consumed, mask_flag)

                    # --- 2. Walk ALL fresh record headers present in this packet ---
                    while pos + 5 <= payload_end:
                        content_type = pkt_bytes[pos]
                        version_major = pkt_bytes[pos + 1]
                        # Valid TLS record header: known content type + version 0x03xx
                        if content_type not in (0x14, 0x15, 0x16, 0x17) or version_major != 0x03:
                            # Desync / non-TLS bytes on a TLS port. If this stream
                            # previously carried valid TLS records, this is almost
                            # certainly ciphertext we lost track of (retransmission,
                            # out-of-order segment) — mask it to honor the "no raw
                            # ciphertext in training data" invariant. If the stream
                            # was never confirmed TLS, leave it visible: plaintext
                            # attack traffic aimed at a TLS port is exactly the kind
                            # of anomaly the model must see.
                            if confirmed_tls_stream:
                                pkt_bytes[pos:payload_end] = b'\x00' * (payload_end - pos)
                            pos = payload_end
                            break
                        record_len = (pkt_bytes[pos + 3] << 8) | pkt_bytes[pos + 4]
                        body_start = pos + 5
                        body_present = min(record_len, max(0, payload_end - body_start))
                        mask_flag = (content_type == 0x17)  # Application Data (encrypted)
                        if mask_flag and body_present > 0:
                            pkt_bytes[body_start: body_start + body_present] = b'\x00' * body_present
                        still_owed = record_len - body_present
                        if stream_tls_state is not None:
                            # Persist entry even at remaining=0: marks this stream as
                            # confirmed-TLS for the desync heuristic above.
                            stream_tls_state[stream_key] = (max(0, still_owed), mask_flag)
                        confirmed_tls_stream = True
                        pos = body_start + body_present

                elif sport in SSH_PORTS or dport in SSH_PORTS:
                    # SSH: everything after the plaintext "SSH-x.x-..." version banner
                    # is either binary KEX or encrypted transport. Mask any payload
                    # that is not the version banner itself — benign encrypted SSH
                    # otherwise trains as irreducible high-entropy noise (and inflates
                    # benign surprise scores at inference). Brute-force patterns
                    # (e.g. hydra) remain visible via banner repetition + packet
                    # rhythm, and are the rate detector's job regardless.
                    if payload_end > payload_start:
                        if bytes(pkt_bytes[payload_start: payload_start + 4]) != b'SSH-':
                            pkt_bytes[payload_start:payload_end] = b'\x00' * (payload_end - payload_start)

        return bytes(pkt_bytes)

    def __iter__(self):
        # SYN flood rate-tracking state (per-worker process, initialised fresh per iteration).
        # A single bare SYN is a normal TCP handshake step; only flag as a flood when the
        # same source IP exceeds SYN_FLOOD_THRESHOLD SYNs within a 1-second sliding window,
        # matching the operational definition in RFC 9293 and CIC-IDS2017 labelling guidelines.
        _syn_tracker: dict = defaultdict(deque)  # {src_ip: deque[float]}
        SYN_FLOOD_THRESHOLD: int = 50            # SYNs / second per source IP
        SYN_WINDOW_SEC: float = 1.0              # sliding-window duration in seconds

        # Cross-packet TLS record continuation state (see _mask_packet_addresses
        # docstring): {(src_ip, dst_ip, sport, dport): remaining_bytes_to_mask}.
        # Fresh per __iter__ call, same lifetime/scope as _syn_tracker above.
        _stream_tls_state: dict = {}

        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        # Prepare buffered anomaly CSV writer (only when labeling is enabled —
        # see label_anomalies in __init__; eval harnesses disable this).
        csv_file = None
        csv_writer = None
        if self.label_anomalies:
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

        import gzip
        pcap_file_obj = gzip.open(self.pcap_path, "rb") if self.pcap_path.endswith(".gz") else open(self.pcap_path, "rb")

        # Bounded shuffle buffer (see class docstring point 5): windows are pooled here,
        # shuffled, and flushed once full, instead of being yielded in raw file order.
        shuffle_pool = []

        def _emit(window_tensor):
            """Queue a window for shuffled emission; flush the pool once it's full."""
            shuffle_pool.append(window_tensor)
            if len(shuffle_pool) >= self.shuffle_buffer_windows:
                random.shuffle(shuffle_pool)
                for w in shuffle_pool:
                    yield w
                shuffle_pool.clear()

        try:
            with RawPcapReader(pcap_file_obj) as pcap_reader:
                buffer = []
                packet_index = 0
                sequence_count = 0
                byte_offset = 0
                
                for packet_data, _metadata in pcap_reader:
                    packet_len = len(packet_data)

                    # 1. Anomaly Labeling & Analysis (optional side channel; skipped
                    #    entirely when label_anomalies=False — big CPU win, and the
                    #    training loss never consumes these labels).
                    if self.label_anomalies:
                        # FIX W4: A bare SYN alone is a normal TCP handshake step.
                        # We only flag TCP_SYN_Flood when the same source IP sends
                        # more than SYN_FLOOD_THRESHOLD SYNs within a 1-second window.
                        #
                        # FIX W6: the window must be measured in PACKET CAPTURE time,
                        # not processing wall-clock time. The old code used
                        # time.monotonic(): a fast disk read squeezed the whole file
                        # into a few wall-clock seconds (labeling nearly every SYN a
                        # "flood"), while slow I/O labeled none. The pcap metadata
                        # carries the real capture timestamp — use it, falling back
                        # to wall clock only if metadata is missing.
                        is_syn_flood = False
                        ts_sec = getattr(_metadata, "sec", None)
                        if ts_sec is None and isinstance(_metadata, tuple) and len(_metadata) > 0:
                            ts_sec = _metadata[0]  # older scapy tuple form (sec, usec, ...)
                        ts_usec = getattr(_metadata, "usec", None)
                        if ts_sec is not None:
                            now = float(ts_sec) + (float(ts_usec) / 1e6 if ts_usec is not None else 0.0)
                        else:
                            now = time.monotonic()
                        try:
                            pkt = Ether(packet_data)
                            if pkt.haslayer(TCP):
                                flags = int(pkt[TCP].flags)
                                # Bare SYN: SYN bit set, ACK bit not set
                                if (flags & 0x02) and not (flags & 0x10):
                                    src_ip = pkt[IP].src if pkt.haslayer(IP) else "unknown"
                                    q = _syn_tracker[src_ip]
                                    q.append(now)
                                    # Evict timestamps that have fallen outside the sliding window
                                    while q and (now - q[0]) > SYN_WINDOW_SEC:
                                        q.popleft()
                                    if len(q) > SYN_FLOOD_THRESHOLD:
                                        is_syn_flood = True
                        except Exception:
                            pass

                        entropy = self._calculate_packet_entropy(packet_data)

                        # Label anomalies in append buffer
                        if is_syn_flood or entropy < 1.0 or (entropy > 7.7 and packet_len > 100):
                            anomaly_type = "TCP_SYN_Flood" if is_syn_flood else "Abnormal_Entropy"
                            csv_writer.writerow([
                                os.path.basename(self.pcap_path),
                                packet_index,
                                byte_offset,
                                anomaly_type,
                                f"{entropy:.4f}"
                            ])

                    # 2. Accumulate packet bytes for partitioned worker
                    if packet_index % num_workers == worker_id:
                        data_to_append = self._mask_packet_addresses(packet_data, stream_tls_state=_stream_tls_state) if self.mask_addresses else packet_data
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
                                    # Tensor cloning to prevent OOM memory pointer leaks; routed
                                    # through the shuffle pool instead of yielding directly.
                                    for emitted in _emit(window.clone()):
                                        yield emitted
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
                            for emitted in _emit(window.clone()):
                                yield emitted
                                sequence_count += 1
                        remainder_start = num_windows * S
                    else:
                        remainder_start = 0

                    # FIX W5: Pad with sentinel value -1 (outside the 0-255 byte range).
                    # FocalLoss(ignore_index=-1) will exclude these positions from every
                    # gradient update, preventing the model wasting capacity on NULL→NULL chains.
                    trailing = flat_bytes[remainder_start:]
                    if len(trailing) > 0:
                        pad_len = W - len(trailing)
                        padded_tensor = torch.cat([trailing, torch.full((pad_len,), -1, dtype=torch.long)])
                        for emitted in _emit(padded_tensor.clone()):
                            yield emitted
                            sequence_count += 1

                # Final flush: emit whatever's left in the shuffle pool (guaranteed < buffer size)
                if shuffle_pool:
                    random.shuffle(shuffle_pool)
                    for w in shuffle_pool:
                        yield w
                        sequence_count += 1
                    shuffle_pool.clear()

            # Atomic Manifest update when loading finishes (only write for main process / worker 0)
            if worker_id == 0:
                self._write_manifest_atomic(sequence_count)
                
        finally:
            pcap_file_obj.close()
            if csv_file is not None:
                csv_file.close()

def get_pcap_dataloader(pcap_path, batch_size=None, num_workers=None, max_sequence_length=8192, stride=None, mask_addresses=True, shuffle_buffer_windows=4096, label_anomalies=True):
    """
    Factory function to create a PyTorch DataLoader for the PCAP streaming dataset.
    Auto-detects the hardware environment to optimize CPU workers and batch size.

    shuffle_buffer_windows: size of the bounded shuffle pool windows are drawn from before
    being emitted (see RawPcapIterableDataset docstring). Larger = better shuffling, more
    RAM. Set to 1 to disable shuffling entirely (strict file order, previous behavior).

    label_anomalies: when False, skips the per-packet anomaly labeling side channel
    (scapy parse + entropy + CSV append). Pass False from evaluation harnesses.
    """
    # Auto-Detector Logic
    import os
    is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

    if batch_size is None:
        batch_size = 64 if is_kaggle else 128

    if num_workers is None:
        num_workers = 4 if is_kaggle else 8

    # CORRECTNESS GUARD: the cross-packet TLS masking state (and any other
    # stream-level state) lives inside a single worker's iterator. With
    # num_workers > 1, packets are partitioned round-robin across workers, so a
    # TLS record's start and its continuation segments land in DIFFERENT worker
    # processes — silently re-breaking the continuation-masking fix and creating
    # a train/eval distribution mismatch (eval always runs single-worker).
    # Masked datasets therefore must be read by a single worker.
    if mask_addresses and num_workers > 1:
        print(f"[GUARD] mask_addresses=True requires a single sequential packet stream; "
              f"forcing num_workers {num_workers} -> 1 (was breaking cross-packet TLS masking state).")
        num_workers = 1

    print(f"[ENV] Configured DataLoader -> Workers: {num_workers} | Batch Size: {batch_size} | Shuffle Buffer: {shuffle_buffer_windows} windows")

    dataset = RawPcapIterableDataset(
        pcap_path,
        max_sequence_length=max_sequence_length,
        stride=stride,
        mask_addresses=mask_addresses,
        shuffle_buffer_windows=shuffle_buffer_windows,
        label_anomalies=label_anomalies
    )
    
    dl_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 8
        dl_kwargs["persistent_workers"] = True

    return DataLoader(dataset, **dl_kwargs)