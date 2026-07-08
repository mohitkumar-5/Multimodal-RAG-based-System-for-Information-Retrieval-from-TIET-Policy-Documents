"""
download_docs.py
----------------
Auto-downloads the PolicyLens PDF corpus zip from Google Drive
and extracts it into the data/ directory.

Usage:
    python download_docs.py
"""

import os
import zipfile
import urllib.request

# ============================================================
# REPLACE THIS with your actual Google Drive direct download link
# Format: https://drive.google.com/uc?export=download&id=FILE_ID
# ============================================================
DRIVE_FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"
ZIP_FILENAME = "tiet_docs.zip"
EXTRACT_DIR = "data"


def download_corpus():
    print("=" * 55)
    print("  PolicyLens — PDF Corpus Downloader")
    print("=" * 55)

    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR)
        print(f"Created directory: {EXTRACT_DIR}/")

    existing = [f for f in os.listdir(EXTRACT_DIR) if f.endswith(".pdf")]
    if existing:
        print(f"\nFound {len(existing)} PDF(s) already in {EXTRACT_DIR}/.")
        answer = input("Re-download and overwrite? [y/N]: ").strip().lower()
        if answer != "y":
            print("Skipping download. Using existing files.")
            return

    if DRIVE_FILE_ID == "YOUR_GOOGLE_DRIVE_FILE_ID_HERE":
        print("\n⚠️  ERROR: Please update DRIVE_FILE_ID in this script.")
        print("   Get the file ID from your Google Drive share link.")
        print("   e.g. https://drive.google.com/file/d/THIS_PART_HERE/view")
        return

    print(f"\n📥 Downloading corpus zip...")
    print(f"   Source: {DOWNLOAD_URL}")

    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, ZIP_FILENAME)
        print(f"   ✅ Downloaded: {ZIP_FILENAME}")
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        print("   Please download manually and place PDFs in the data/ folder.")
        return

    print(f"\n📂 Extracting into {EXTRACT_DIR}/...")
    try:
        with zipfile.ZipFile(ZIP_FILENAME, "r") as zf:
            zf.extractall(EXTRACT_DIR)
        print(f"   ✅ Extracted successfully.")
        os.remove(ZIP_FILENAME)
        print(f"   🗑️  Removed temporary zip file.")
    except zipfile.BadZipFile:
        print("   ❌ The downloaded file is not a valid zip.")
        print("   Google Drive may have served an HTML warning page.")
        print("   Please download the zip manually for large files.")
        os.remove(ZIP_FILENAME)
        return

    pdf_count = len([f for f in os.listdir(EXTRACT_DIR) if f.endswith(".pdf")])
    print(f"\n✅ Done! {pdf_count} PDF file(s) are now in {EXTRACT_DIR}/")
    print("   You can now run the server: uvicorn app.main:app --reload")


if __name__ == "__main__":
    download_corpus()
