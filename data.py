import requests
from tqdm import tqdm
import zipfile
import os

# Corrected Zenodo link for the Choral Singing Dataset
CHORALSET_URL = "https://zenodo.org/records/2649950/files/ChoralSingingDataset.zip?download=1"
OUTPUT_ZIP = "ChoralSingingDataset.zip"
EXTRACT_DIR = "ChoralSingingDataset"

def download_choralsinging_dataset():
    if os.path.exists(EXTRACT_DIR):
        print(f"[✓] '{EXTRACT_DIR}' folder already exists. Skipping download.")
        return

    print(f"[↓] Downloading Choral Singing Dataset...")
    response = requests.get(CHORALSET_URL, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with open(OUTPUT_ZIP, "wb") as f, tqdm(
        desc=OUTPUT_ZIP,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    print(f"[✓] Download complete. Extracting...")
    with zipfile.ZipFile(OUTPUT_ZIP, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    print(f"[✓] Extraction complete. Data saved in: {EXTRACT_DIR}/")
    os.remove(OUTPUT_ZIP)

if __name__ == "__main__":
    download_choralsinging_dataset()
