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
echo "To pull public baselines directly to the cluster storage, execute the following commands:"
echo ""
echo "1. CIC-IDS2017 dataset (Canadian Institute for Cybersecurity):"
echo "   wget -O data/cic-ids2017/dataset.zip http://205.174.165.80/CICDataset/CIC-IDS2017/Dataset/PCAPs/Wednesday-workingHours.pcap.zip"
echo "   unzip data/cic-ids2017/dataset.zip -d data/cic-ids2017/"
echo ""
echo "2. MACCDC capture dataset:"
echo "   wget -O data/maccdc/maccdc2012_00001.pcap.gz https://download.netresec.com/pcap/maccdc-2012/maccdc2012_00001.pcap.gz"
echo "   gunzip data/maccdc/maccdc2012_00001.pcap.gz"
echo ""
echo "3. CTU-13 dataset:"
echo "   wget -O data/ctu-13/ctu13_botnet.pcap https://mcfp.felk.cvut.cz/publicFiles/datasets/CTU-13-Dataset/CTU-13-1/capture20110810.pcap"
echo ""
echo "Please make sure you have sufficient disk space (typically 20GB-50GB for full captures) before launching downloads."
