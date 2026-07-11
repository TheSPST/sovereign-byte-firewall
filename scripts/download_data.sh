#!/bin/bash
# Dataset Downloader & Preparer for Sovereign Byte-Level Firewall.
# Creates data directories and provides instructions/commands to download public PCAP baselines.
#
# Usage:
#   bash scripts/download_data.sh

# Create destination directories
mkdir -p data/cic-ids2017
mkdir -p data/maccdc
mkdir -p data/ctu-13

echo "=================================================="
# Local Dev fallback setup
if [ -f "local_test.pcap" ]; then
    echo "Local test PCAP found! Creating a symlink for local validation..."
    ln -sf ../../local_test.pcap data/cic-ids2017/cic_ids.pcap
    echo "Created symlink: data/cic-ids2017/cic_ids.pcap -> local_test.pcap"
fi
echo "=================================================="

echo "Data directories initialized successfully."
echo "To pull public baselines, execute the following commands:"
echo ""
echo "1. CIC-IDS2017 dataset (Canadian Institute for Cybersecurity)."
echo ""
echo "   IMPORTANT: Monday-WorkingHours.pcap is the PURE BENIGN day (no attacks"
echo "   mixed in) — confirmed on unb.ca/cic/datasets/ids-2017.html. This is the"
echo "   correct file for Stage 1 unsupervised training, which is designed to"
echo "   learn ONLY 'normal' byte-level protocol grammar. Wednesday/Tuesday/"
echo "   Thursday/Friday all contain attacks mixed into the same capture and"
echo "   should be reserved for evaluate_zero_day.py, never for Stage 1 training."
echo ""
echo "   IMPORTANT: The old CIC direct-download host (205.174.165.80) now sits"
echo "   behind a request form on unb.ca for browser access; direct wget links"
echo "   against it may or may not still work depending on the day. The"
echo "   reliable path below uses a verified Hugging Face mirror that hosts the"
echo "   original raw PCAPs (not just flow-feature CSVs) via Git LFS."
echo ""
echo "   # Requires: pip install -U 'huggingface_hub[cli]'"
echo "   huggingface-cli download bvsam/cic-ids-2017 --repo-type dataset \\"
echo "       --include 'pcap/Monday*' \\"
echo "       --local-dir data/cic-ids2017"
echo ""
echo "   # Optional — pull the attack days too, for a LARGER labeled eval set"
echo "   # (do not train Stage 1 on these, evaluate_zero_day.py only):"
echo "   huggingface-cli download bvsam/cic-ids-2017 --repo-type dataset \\"
echo "       --include 'pcap/Tuesday*' 'pcap/Wednesday*' 'pcap/Thursday*' 'pcap/Friday*' \\"
echo "       --local-dir data/cic-ids2017"
echo ""
echo "   NOTE: Do NOT use the Kaggle dataset 'chethuhn/network-intrusion-dataset' —"
echo "   it only contains CICFlowMeter-extracted CSV flow features (already"
echo "   aggregated statistics like packet counts and IATs), not raw packet"
echo "   bytes. This project's byte-level model needs raw .pcap files."
echo ""
echo "2. Split Monday's pcap into a training slice and a held-out validation"
echo "   slice BEFORE uploading anywhere (requires Wireshark's 'editcap', or"
echo "   substitute tcpdump -r/-w with packet-count ranges). Never train and"
echo "   validate on overlapping bytes of the same file."
echo "   editcap -r data/cic-ids2017/pcap/Monday-WorkingHours.pcap \\"
echo "       data/cic-ids2017/monday_train.pcap 1-3500000"
echo "   editcap -r data/cic-ids2017/pcap/Monday-WorkingHours.pcap \\"
echo "       data/cic-ids2017/monday_val.pcap 3500001-9999999999"
echo "   (Adjust packet ranges to your actual packet count; check with"
echo "   'capinfos data/cic-ids2017/pcap/Monday-WorkingHours.pcap' first — aim"
echo "   for roughly an 85/15 train/val split.)"
echo ""
echo "3. MACCDC capture dataset:"
echo "   wget -O data/maccdc/maccdc2012_00001.pcap.gz https://download.netresec.com/pcap/maccdc-2012/maccdc2012_00001.pcap.gz"
echo "   gunzip data/maccdc/maccdc2012_00001.pcap.gz"
echo ""
echo "4. CTU-13 dataset:"
echo "   wget -O data/ctu-13/ctu13_botnet.pcap https://mcfp.felk.cvut.cz/publicFiles/datasets/CTU-13-Dataset/CTU-13-1/capture20110810.pcap"
echo ""
echo "Please make sure you have sufficient disk space (Monday alone is ~11GB;"
echo "all 5 days together are ~51GB) before launching downloads."
