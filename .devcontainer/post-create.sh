#!/bin/bash
# Runs once when the Codespace is created. No checkpoint is downloaded.
set -euo pipefail
sudo apt-get update
sudo apt-get install -y build-essential python3
echo "== native plan =="
make -C native test
echo "== native range reader =="
make -C native stream
echo "== engine =="
make -j
echo "== weightless gates =="
make test
echo
echo "Codespace checks finished."
echo "The 1.67 TB checkpoint does not fit this disk. See HANDSET.md."
