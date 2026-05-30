"""Download the OfficeQA transformed text files from Hugging Face."""

import os
import sys
from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN")
if not token:
    print("ERROR: HF_TOKEN environment variable is not set.")
    print("Run:  $env:HF_TOKEN = 'hf_...'   (PowerShell)")
    sys.exit(1)

print("Downloading transformed TXT files (~460MB)...")
local_dir = snapshot_download(
    repo_id="databricks/officeqa",
    repo_type="dataset",
    allow_patterns="treasury_bulletins_parsed/transformed/*.txt",
    local_dir="./officeqa-corpus",
    token=token,
)
print(f"\nDone! Files saved to: {local_dir}")
print("Path: officeqa-corpus/treasury_bulletins_parsed/transformed/")
