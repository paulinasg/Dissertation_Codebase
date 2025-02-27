import os
import requests
from tqdm import tqdm
import time

def download_file(url, destination_folder, filename):
    """
    Download a file from the given URL to the specified destination folder with the given filename.
    Shows a progress bar during download.
    """
    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Full path for the file
    file_path = os.path.join(destination_folder, filename)
    
    # Check if file already exists
    if os.path.exists(file_path):
        print(f"File {filename} already exists. Skipping download.")
        return
    
    # Stream the download to handle large files
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors
    
    # Get the total file size if available
    total_size = int(response.headers.get('content-length', 0))
    
    # Download the file with progress bar
    with open(file_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"Downloaded {filename} successfully!")

def download_selected_files(base_url, files_to_download, destination_folder="downloads"):
    """
    Download selected files from the dataset.
    
    Args:
        base_url: The base URL for the dataset
        files_to_download: List of filenames to download
        destination_folder: Folder to save downloaded files
    """
    for filename in files_to_download:
        file_url = f"{base_url}/{filename}"
        print(f"Downloading {filename}...")
        
        try:
            download_file(file_url, destination_folder, filename)
            # Add a small delay between downloads to avoid overwhelming the server
            time.sleep(1)
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Replace this with your custom URL that was emailed to you
    base_url = "https://4d-dress.ait.ethz.ch/download.php?dt=def5020061cd969db17542a2395172e2884c78be8870b1e866c5ff72caf9050c745f6fbfe4637d60f45fd02c3e18c49a0d0ce2cb04e2e53290f1e1d52788a74c5fd16e4a641deddb9c723ad27afa22cabc6d4ea8d3a6e569e744e2a1df26c207a8ff71d0bb4377f7a9df19866a5cd167f77b"
    
    # List of files you want to download (from the image)
    files_to_download = [
        "00137_Outer_1.tar.gz",
        "00140_Outer_1.tar.gz",
        "00147_Outer.tar.gz",
        "00148_Outer.tar.gz",
        "00149_Outer_1.tar.gz",
        "00151_Outer.tar.gz",
        "00152_Outer_1.tar.gz",
        "00154_Outer_1.tar.gz",
        "00156_Outer.tar.gz",
        "00160_Outer.tar.gz",
        "00163_Outer.tar.gz",
        "00167_Outer.tar.gz",
        "00168_Outer_1.tar.gz",
        "00169_Outer.tar.gz",
        "00170_Outer.tar.gz",
        "00174_Outer.tar.gz",
        "00175_Outer_1.tar.gz",
        "00176_Outer.tar.gz",
        "00179_Outer.tar.gz",
        "00180_Outer.tar.gz",
        "00185_Outer_1.tar.gz",
        "00187_Outer.tar.gz",
        "00188_Outer.tar.gz",
        "00190_Outer.tar.gz",
        "00191_Outer.tar.gz",
        # Add more filenames as needed
    ]
    
    # Set destination folder
    destination_folder = "4D-DRESS_downloads"
    
    # Download the selected files
    download_selected_files(base_url, files_to_download, destination_folder)