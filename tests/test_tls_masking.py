"""
Unit tests for the cross-packet TLS / SSH payload masking in
src.dataloader.RawPcapIterableDataset._mask_packet_addresses.

These tests are pure byte-level: they build synthetic Ethernet/IPv4/TCP frames
and call the masking function directly, so they run WITHOUT torch installed
(torch is stubbed if missing — the masking path never touches it). This keeps
the test runnable on thin CI/sandbox environments as well as Kaggle/AIKosh.

Covered bug classes (see _mask_packet_addresses docstring, FIX history):
  (a) App-data record spanning multiple packets -> continuation masked
  (b) Multiple records per packet, second spans onward -> continuation masked
  (c) Handshake record spanning packets -> continuation consumed, NOT masked
  (d) Desync on a confirmed TLS stream -> masked; plaintext on 443 from a
      never-confirmed stream -> left visible
  (e) SSH: version banner visible, other payload masked
  (f) Non-443 TLS ports (e.g. 8443) now masked
"""
import sys
import types
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# --- torch stub (only needed when torch isn't installed; masking uses no torch) ---
try:
    import torch  # noqa: F401
except ImportError:
    torch_stub = types.ModuleType("torch")
    utils_stub = types.ModuleType("torch.utils")
    data_stub = types.ModuleType("torch.utils.data")

    class _IterableDataset:  # minimal stand-ins
        pass

    data_stub.IterableDataset = _IterableDataset
    data_stub.DataLoader = object
    data_stub.get_worker_info = lambda: None
    utils_stub.data = data_stub
    torch_stub.utils = utils_stub
    sys.modules["torch"] = torch_stub
    sys.modules["torch.utils"] = utils_stub
    sys.modules["torch.utils.data"] = data_stub

from src.dataloader import RawPcapIterableDataset  # noqa: E402


SRC_IP = bytes([10, 0, 0, 1])
DST_IP = bytes([10, 0, 0, 2])


def build_packet(payload, sport=50000, dport=443, src_ip=SRC_IP, dst_ip=DST_IP):
    """Minimal Ethernet + IPv4 + TCP frame around `payload`."""
    eth = bytes([0xAA] * 6 + [0xBB] * 6) + b"\x08\x00"
    ip = bytearray(20)
    ip[0] = 0x45                      # version 4, IHL 5
    ip[9] = 6                         # protocol TCP
    ip[12:16] = src_ip
    ip[16:20] = dst_ip
    tcp = bytearray(20)
    tcp[0:2] = sport.to_bytes(2, "big")
    tcp[2:4] = dport.to_bytes(2, "big")
    tcp[12] = 0x50                    # data offset 5 words = 20 bytes, no options
    return eth + bytes(ip) + bytes(tcp) + payload


PAYLOAD_OFFSET = 14 + 20 + 20  # eth + ipv4 + tcp


def tls_record(content_type, body, declared_len=None):
    """TLS record header + (possibly partial) body."""
    length = declared_len if declared_len is not None else len(body)
    return bytes([content_type, 0x03, 0x03]) + length.to_bytes(2, "big") + body


def mask(packet, state):
    # _mask_packet_addresses only reads module-level constants + args, so any
    # object works as `self` — no need to construct a dataset (which would
    # require a real pcap file on disk).
    return RawPcapIterableDataset._mask_packet_addresses(object(), packet, stream_tls_state=state)


def payload_of(masked_packet):
    return masked_packet[PAYLOAD_OFFSET:]


def test_appdata_continuation_is_masked():
    """(a) The original bug: record bigger than one segment."""
    state = {}
    body_part1 = bytes([0x41] * 100)
    pkt1 = build_packet(tls_record(0x17, body_part1, declared_len=300), )
    out1 = mask(pkt1, state)
    # header (5 bytes) survives, body masked
    assert payload_of(out1)[:5] == bytes([0x17, 0x03, 0x03, 0x01, 0x2C])
    assert payload_of(out1)[5:] == b"\x00" * 100

    # continuation packet: 200 remaining ciphertext bytes, starts mid-record
    pkt2 = build_packet(bytes([0x99] * 200))
    out2 = mask(pkt2, state)
    assert payload_of(out2) == b"\x00" * 200, "continuation ciphertext must be masked"
    key = next(iter(state))
    assert state[key] == (0, True)


def test_second_record_in_same_packet_spans_packets():
    """(b) Only the first record header used to be parsed."""
    state = {}
    rec_a = tls_record(0x17, bytes([0x41] * 10))                      # complete
    rec_b = tls_record(0x17, bytes([0x42] * 20), declared_len=100)    # spans onward
    pkt1 = build_packet(rec_a + rec_b)
    out1 = mask(pkt1, state)
    p = payload_of(out1)
    assert p[5:15] == b"\x00" * 10          # record A body masked
    assert p[20:40] == b"\x00" * 20         # record B partial body masked
    # 80 bytes of record B still owed on this stream
    key = next(iter(state))
    assert state[key] == (80, True)

    pkt2 = build_packet(bytes([0x77] * 80))
    out2 = mask(pkt2, state)
    assert payload_of(out2) == b"\x00" * 80


def test_handshake_continuation_consumed_not_masked():
    """(c) Certificate chains span packets but are NOT encrypted payload."""
    state = {}
    pkt1 = build_packet(tls_record(0x16, bytes([0x0B] * 50), declared_len=200))
    out1 = mask(pkt1, state)
    assert payload_of(out1)[5:] == bytes([0x0B] * 50), "handshake body must stay visible"

    # continuation of the handshake record: 150 bytes; first byte 0x17 by bad
    # luck — the old code would have misread it as a fresh app-data header.
    cont = bytes([0x17]) + bytes([0x0C] * 149)
    pkt2 = build_packet(cont)
    out2 = mask(pkt2, state)
    assert payload_of(out2) == cont, "handshake continuation must not be masked"
    key = next(iter(state))
    assert state[key] == (0, False)


def test_desync_masked_only_on_confirmed_tls_stream():
    """(d) Unparseable payload on 443: mask iff stream previously spoke TLS."""
    # Fresh stream, plaintext on 443 (e.g. an exploit) -> stays visible
    state = {}
    plain = b"GET /shell?cmd=id HTTP/1.1\r\n"
    out = mask(build_packet(plain), state)
    assert payload_of(out) == plain
    assert state == {}

    # Confirmed TLS stream, then garbage (out-of-order ciphertext) -> masked
    state = {}
    mask(build_packet(tls_record(0x17, bytes([0x41] * 10))), state)   # confirms stream
    garbage = bytes([0xDE, 0xAD] * 30)
    out2 = mask(build_packet(garbage), state)
    assert payload_of(out2) == b"\x00" * 60


def test_ssh_banner_visible_rest_masked():
    """(e) SSH payload masking."""
    state = {}
    banner = b"SSH-2.0-OpenSSH_8.9\r\n"
    out1 = mask(build_packet(banner, dport=22), state)
    assert payload_of(out1) == banner

    binary = bytes(range(64))
    out2 = mask(build_packet(binary, dport=22), state)
    assert payload_of(out2) == b"\x00" * 64


def test_non_443_tls_port_masked():
    """(f) Port coverage widened beyond 443."""
    state = {}
    pkt = build_packet(tls_record(0x17, bytes([0x41] * 30)), dport=8443)
    out = mask(pkt, state)
    assert payload_of(out)[5:] == b"\x00" * 30


def test_addresses_still_masked():
    """Regression: MAC + IP blanking untouched by the TLS rewrite."""
    out = mask(build_packet(b""), {})
    assert out[0:12] == b"\x00" * 12                     # MACs
    assert out[14 + 12: 14 + 20] == b"\x00" * 8          # IPs


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} masking tests passed.")
