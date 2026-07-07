import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from scapy.utils import RawPcapReader

class RawPcapIterableDataset(IterableDataset):
    """
    A PyTorch IterableDataset that streams raw bytes from a PCAP file packet-by-packet.
    It yields sequences of a fixed length (max_sequence_length), zero-padded at the end.
    Supports multi-process loading by partitioning packets among workers.
    """
    def __init__(self, pcap_path, max_sequence_length=8192):
        super().__init__()
        self.pcap_path = pcap_path
        self.max_sequence_length = max_sequence_length

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        try:
            with RawPcapReader(self.pcap_path) as pcap_reader:
                current_sequence = []
                packet_index = 0
                
                for packet_data, _metadata in pcap_reader:
                    if packet_index % num_workers == worker_id:
                        for byte in packet_data:
                            current_sequence.append(byte)
                            if len(current_sequence) == self.max_sequence_length:
                                yield torch.tensor(current_sequence, dtype=torch.long)
                                current_sequence = []
                    packet_index += 1
                
                # Yield any remaining trailing bytes padded with 0x00
                if current_sequence:
                    remainder = self.max_sequence_length - len(current_sequence)
                    current_sequence.extend([0] * remainder)
                    yield torch.tensor(current_sequence, dtype=torch.long)
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"Target PCAP file not found at: {self.pcap_path}")

def get_pcap_dataloader(pcap_path, batch_size=32, num_workers=0, max_sequence_length=8192):
    """
    Factory function to create a PyTorch DataLoader for the PCAP streaming dataset.
    Defaults to num_workers=0 (single-process) for robust local prototyping.
    """
    dataset = RawPcapIterableDataset(pcap_path, max_sequence_length=max_sequence_length)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
