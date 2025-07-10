import requests
import zipfile
import os
from tqdm import tqdm

# URL and target paths
url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
zip_path = "maestro-v3.0.0.zip"
extract_path = "maestro-v3.0.0"

print("🔽 Starting download...")

# Request with streamed chunks and Content-Length for progress
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    total_size = int(r.headers.get("Content-Length", 0))
    block_size = 8192
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")

    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=block_size):
            f.write(chunk)
            progress_bar.update(len(chunk))

    progress_bar.close()

print(f"✅ Download complete. Saved to '{zip_path}'.")

# Extract the zip file
print("📦 Extracting files...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)
print(f"✅ Extraction complete. Files extracted to '{extract_path}'.")

# Optional: show size of extracted folder
def get_dir_size(path):
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size

size_bytes = get_dir_size(extract_path)
print(f"📊 Extracted dataset size: {size_bytes / 1e9:.2f} GB")
