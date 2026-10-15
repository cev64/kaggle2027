#!/bin/bash
# Downloads the NFL Big Data Bowl 2027 competition data into data/raw/.
# Requires a Kaggle API token: set KAGGLE_API_TOKEN or save ~/.kaggle/access_token,
# and accept the competition rules on kaggle.com first.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
kaggle competitions download -c nfl-big-data-bowl-2027 -p data/raw
cd data/raw && unzip -o -q nfl-big-data-bowl-2027.zip && rm nfl-big-data-bowl-2027.zip
ls -la
