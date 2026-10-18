import urllib.request
import os

url = "https://ember.elastic.co/ember_dataset_2018_2.tar.bz2"
filename = "data/ember_2018/ember_dataset_2018_2.tar.bz2"

os.makedirs("data/ember_2018", exist_ok=True)

print("Downloading EMBER 2018 (1.7 GB)...")
print("This may take 20-40 minutes. Do not close this window.")

urllib.request.urlretrieve(url, filename)

print(f"\nDone! File size: {os.path.getsize(filename):,} bytes")
print("Extract this file with 7-Zip or WinRAR, then run your EDA notebook.")