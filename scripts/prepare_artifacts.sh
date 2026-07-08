#!/bin/bash
# Script to package project source code for cluster deployment.
# Safely ignores heavy files like virtual environments, local PCAPs, and temporary logs/checkpoints.
#
# Usage:
#   bash scripts/prepare_artifacts.sh

OUTPUT_ZIP="sovereign_byte_firewall.zip"

echo "=================================================="
echo " Preparing clean deployment package for AI Kosh..."
echo "=================================================="

# Check if zip is installed
if ! command -v zip &> /dev/null; then
    echo "ERROR: 'zip' utility is not installed on this Mac. Please run 'brew install zip' or use Finder."
    exit 1
fi

# Clean up any existing zip archive
if [ -f "$OUTPUT_ZIP" ]; then
    echo "Removing old $OUTPUT_ZIP..."
    rm "$OUTPUT_ZIP"
fi

# Create a clean zip archive excluding local runtime folders
zip -r "$OUTPUT_ZIP" . \
    -x "*.venv*" \
    -x "*venv*" \
    -x "*env*" \
    -x "*__pycache__*" \
    -x "*.git*" \
    -x "*checkpoints*" \
    -x "*test_checkpoints*" \
    -x "*logs*" \
    -x "*results*" \
    -x "*.pcap" \
    -x "*.zip" \
    -x "*.DS_Store" \
    -x "*.tmp*"

echo "=================================================="
if [ -f "$OUTPUT_ZIP" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_ZIP" | cut -f1)
    echo " SUCCESS: Deployment archive created: $OUTPUT_ZIP ($FILE_SIZE)"
    echo " Upload this single file to the AI Kosh notebook file-copy interface."
else
    echo " ERROR: Failed to create zip archive."
fi
echo "=================================================="
