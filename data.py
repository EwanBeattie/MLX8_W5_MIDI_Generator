import requests
import zipfile
import os

# URL and target paths
url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
zip_path = "maestro-v3.0.0.zip"
extract_path = "maestro-v3.0.0"

# Download the zip file (streamed)
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

print("✅ Download complete.")

# Extract the zip file
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

print("✅ Extraction complete.")
