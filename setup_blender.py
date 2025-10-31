# setup_blender.py
import sys
import os
import requests
import zipfile
import tarfile
import platform
from tqdm import tqdm

# --- Configuration ---
BLENDER_VERSION = "4.1.1"
INSTALL_DIR = os.path.abspath("bin")
BLENDER_DIR = os.path.join(INSTALL_DIR, "blender")

# URLs for different platforms
URLS = {
    "Windows": f"https://download.blender.org/release/Blender{BLENDER_VERSION[:3]}/blender-{BLENDER_VERSION}-windows-x64.zip",
    "Linux": f"https://download.blender.org/release/Blender{BLENDER_VERSION[:3]}/blender-{BLENDER_VERSION}-linux-x64.tar.xz",
    "Darwin": f"https://download.blender.org/release/Blender{BLENDER_VERSION[:3]}/blender-{BLENDER_VERSION}-macos-x64.dmg", # DMG is tricky, let's target zip if available or warn user
}

def get_blender_executable_path():
    """Returns the expected path to the Blender executable."""
    system = platform.system()
    if system == "Windows":
        return os.path.join(BLENDER_DIR, f"blender-{BLENDER_VERSION}-windows-x64", "blender.exe")
    elif system == "Linux":
        return os.path.join(BLENDER_DIR, f"blender-{BLENDER_VERSION}-linux-x64", "blender")
    elif system == "Darwin": # macOS
        # This is more complex due to .app structure inside a DMG
        return f"Blender executable path for macOS DMG requires manual setup. Look inside {BLENDER_DIR}."
    return None

def download_file(url, dest_path):
    """Downloads a file with a progress bar."""
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=dest_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)

def extract_archive(filepath, dest_dir):
    """Extracts a .zip or .tar.xz file."""
    print(f"Extracting {filepath} to {dest_dir}...")
    if filepath.endswith(".zip"):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
    elif filepath.endswith(".tar.xz"):
        with tarfile.open(filepath, 'r:xz') as tar_ref:
            tar_ref.extractall(dest_dir)
    else:
        print(f"Unsupported archive format: {filepath}. Manual extraction required.")
        return
    print("Extraction complete.")

def main():
    """Main setup function."""
    print("--- Setting up Blender Dependency ---")
    
    blender_exe = get_blender_executable_path()
    if blender_exe and os.path.exists(blender_exe):
        print(f"Blender executable already found at: {blender_exe}")
        print("Setup complete.")
        return

    system = platform.system()
    if system not in URLS:
        print(f"Unsupported OS: {system}. Please install Blender {BLENDER_VERSION} manually.")
        sys.exit(1)
        
    if system == "Darwin":
        print("macOS DMG download detected. Automatic extraction is not supported.")
        print("Please download Blender manually from blender.org and place it in the project directory.")
        sys.exit(1)

    url = URLS[system]
    archive_name = url.split('/')[-1]
    
    os.makedirs(INSTALL_DIR, exist_ok=True)
    archive_path = os.path.join(INSTALL_DIR, archive_name)

    if not os.path.exists(archive_path):
        download_file(url, archive_path)
    
    extract_archive(archive_path, BLENDER_DIR)
    
    if blender_exe and os.path.exists(blender_exe):
        print(f"\nBlender is ready to use at: {blender_exe}")
        print("You can now run your generation script.")
    else:
        print("\nSetup finished, but could not verify blender executable path. Please check the 'bin' directory.")

if __name__ == "__main__":
    main()