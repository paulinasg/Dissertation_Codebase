import os
import time
import urllib.parse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import threading
import glob

class ETHZAdvancedDownloader:
    def __init__(self, base_url, token, download_folder="4D-DRESS_downloads"):
        self.base_url = base_url
        self.token = token
        
        # Get absolute path for download folder
        self.download_folder = os.path.abspath(download_folder)
        
        # Create download folder if it doesn't exist
        if not os.path.exists(self.download_folder):
            os.makedirs(self.download_folder)
        
        print(f"Files will be downloaded to: {self.download_folder}")
        
        # Configure Chrome options
        self.chrome_options = Options()
        self.chrome_options.add_experimental_option("prefs", {
            "download.default_directory": self.download_folder,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        })
        
        # Set up some additional options to make downloads more reliable
        # Uncomment if you want to see what's happening
        # self.chrome_options.add_argument("--start-maximized")
        
        # To run in headless mode, uncomment these lines
        # self.chrome_options.add_argument("--headless=new")
        # self.chrome_options.add_argument("--window-size=1920,1080")
        # self.chrome_options.add_argument("--disable-gpu")
        
        # Add more browser-like characteristics
        self.chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        self.chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.chrome_options.add_experimental_option("useAutomationExtension", False)
        
        # Initialize the browser
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=self.chrome_options
        )
        
        # Modify navigator.webdriver property to prevent detection
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        # Set page load timeout
        self.driver.set_page_load_timeout(60)
        
        # State tracking
        self.download_in_progress = False
        self.current_file = None
    
    def _build_download_url(self, filename):
        """Build the download URL for a specific file."""
        encoded_filename = urllib.parse.quote(filename)
        return f"{self.base_url}?dt={self.token}/{encoded_filename}"
    
    def _is_file_downloaded(self, filename, timeout=60):
        """Check if a file has been downloaded and is complete."""
        start_time = time.time()
        file_path = os.path.join(self.download_folder, filename)
        
        while time.time() - start_time < timeout:
            # Check for the exact file
            if os.path.exists(file_path):
                # Make sure it's not a temporary file
                if not os.path.exists(file_path + ".crdownload") and not os.path.exists(file_path + ".tmp"):
                    return True
            
            # Check for Chrome's temporary download files
            temp_files = glob.glob(os.path.join(self.download_folder, "*.crdownload"))
            temp_files.extend(glob.glob(os.path.join(self.download_folder, "*.tmp")))
            
            # If no download is in progress but we've waited at least 10 seconds, return False
            if not temp_files and time.time() - start_time > 10:
                return False
                
            time.sleep(1)
        
        # Timeout reached
        return False
    
    def _wait_for_download_complete(self, filename, timeout=3600):
        """Wait for a file download to complete, with progress monitoring."""
        file_path = os.path.join(self.download_folder, filename)
        start_time = time.time()
        last_size = 0
        last_update_time = time.time()
        
        print(f"Waiting for download to complete: {filename}")
        
        while time.time() - start_time < timeout:
            if os.path.exists(file_path):
                current_size = os.path.getsize(file_path)
                
                # Check if the file is growing
                if current_size > last_size:
                    size_diff = current_size - last_size
                    time_diff = time.time() - last_update_time
                    
                    # Only update if meaningful time has passed
                    if time_diff > 5:
                        speed = size_diff / time_diff / (1024*1024)  # MB/s
                        progress_gb = current_size / (1024*1024*1024)
                        
                        print(f"Downloading {filename}: {progress_gb:.2f} GB ({speed:.2f} MB/s)")
                        
                        last_size = current_size
                        last_update_time = time.time()
                
                # Check for download completion
                # Look for temporary files
                temp_file = file_path + ".crdownload"
                temp_file2 = file_path + ".tmp"
                
                if (not os.path.exists(temp_file) and 
                    not os.path.exists(temp_file2) and
                    time.time() - last_update_time > 10):
                    # No temp files and no size change for 10 seconds
                    file_size = os.path.getsize(file_path) / (1024*1024*1024)
                    print(f"Download completed: {filename} ({file_size:.2f} GB)")
                    self.download_in_progress = False
                    self.current_file = None
                    return True
            
            # If nothing is happening for too long (no progress for 5 minutes), consider it failed
            if time.time() - last_update_time > 300 and last_size > 0:
                print(f"Download seems stalled for {filename}. No progress for 5 minutes.")
                self.download_in_progress = False
                self.current_file = None
                return False
                
            time.sleep(5)  # Check every 5 seconds
        
        # Timeout reached
        print(f"Download timeout for {filename}")
        self.download_in_progress = False
        self.current_file = None
        return False
    
    def download_file(self, filename):
        """Download a single file."""
        file_path = os.path.join(self.download_folder, filename)
        
        # Check if file already exists
        if os.path.exists(file_path):
            size_gb = os.path.getsize(file_path) / (1024*1024*1024)
            print(f"File {filename} already exists ({size_gb:.2f} GB). Skipping.")
            return True
        
        # Generate download URL
        download_url = self._build_download_url(filename)
        
        print(f"Attempting to download: {filename}")
        print(f"URL: {download_url}")
        
        # Set download state
        self.download_in_progress = True
        self.current_file = filename
        
        try:
            # Navigate to the download URL
            self.driver.get(download_url)
            
            # Wait a moment for any redirects
            time.sleep(5)
            
            # Check for errors in the page title or content
            page_title = self.driver.title.lower()
            page_source = self.driver.page_source.lower()
            
            if "forbidden" in page_title or "error" in page_title or "404" in page_title:
                print(f"Error page detected for {filename}: {page_title}")
                self.download_in_progress = False
                self.current_file = None
                return False
            
            if "forbidden" in page_source or "access denied" in page_source:
                print(f"Access denied for {filename}")
                self.download_in_progress = False
                self.current_file = None
                return False
            
            # If we got here, the download should have started automatically
            # Wait for the download to complete
            result = self._wait_for_download_complete(filename)
            return result
            
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
            self.download_in_progress = False
            self.current_file = None
            return False
    
    def download_files(self, filenames):
        """Download a list of files sequentially."""
        results = {}
        
        for filename in filenames:
            try:
                print(f"\n{'='*50}\nProcessing file: {filename}\n{'='*50}")
                success = self.download_file(filename)
                results[filename] = success
                
                # Add a small delay between downloads
                time.sleep(3)
            except Exception as e:
                print(f"Unexpected error processing {filename}: {str(e)}")
                results[filename] = False
        
        # Print summary
        print("\n\n==== DOWNLOAD SUMMARY ====")
        successful = [f for f, result in results.items() if result]
        failed = [f for f, result in results.items() if not result]
        
        print(f"Successfully downloaded: {len(successful)}/{len(filenames)}")
        for f in successful:
            file_path = os.path.join(self.download_folder, f)
            size_gb = os.path.getsize(file_path) / (1024*1024*1024) if os.path.exists(file_path) else 0
            print(f"  ✓ {f} ({size_gb:.2f} GB)")
        
        if failed:
            print(f"\nFailed downloads: {len(failed)}/{len(filenames)}")
            for f in failed:
                print(f"  ✗ {f}")
    
    def close(self):
        """Close the browser."""
        if hasattr(self, 'driver') and self.driver:
            self.driver.quit()
            print("Browser closed.")

def main():
    # URL and token
    base_url = "https://4d-dress.ait.ethz.ch/download.php"
    token = "def5020061cd969db17542a2395172e2884c78be8870b1e866c5ff72caf9050c745f6fbfe4637d60f45fd02c3e18c49a0d0ce2cb04e2e53290f1e1d52788a74c5fd16e4a641deddb9c723ad27afa22cabc6d4ea8d3a6e569e744e2a1df26c207a8ff71d0bb4377f7a9df19866a5cd167f77b"
    
    # Files to download
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
        "00191_Outer.tar.gz"
    ]
    
    downloader = None
    try:
        # Initialize the downloader
        downloader = ETHZAdvancedDownloader(base_url, token)
        
        # Download the files
        downloader.download_files(files_to_download)
        
    except KeyboardInterrupt:
        print("\nDownload process interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    finally:
        # Make sure to close the browser
        if downloader:
            downloader.close()

if __name__ == "__main__":
    main()