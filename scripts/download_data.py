import urllib.request
import os

def download_real_cfd_data():
    # URL to the raw .mat file from the original PINNs repository
    url = "https://github.com/maziarraissi/PINNs/raw/master/main/Data/cylinder_nektar_wake.mat"
    
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", "cylinder_nektar_wake.mat")
    
    print(f"[*] Downloading real CFD dataset...")
    print(f"[*] Source: {url}")
    
    try:
        urllib.request.urlretrieve(url, filepath)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"[+] Download complete! Saved to {filepath}")
        print(f"[*] File size: {size_mb:.2f} MB (Should be ~30+ MB)")
    except Exception as e:
        print(f"[!] Download failed: {e}")

if __name__ == "__main__":
    download_real_cfd_data()