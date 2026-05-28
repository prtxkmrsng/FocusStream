import urllib.request
import os

# URLs pointing directly to the raw files in OpenCV's official repository
urls = {
    "haarcascade_frontalface_default.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
    "haarcascade_eye.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
}

for filename, url in urls.items():
    print(url)
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Successfully saved {filename}")
    else:
        print(f"{filename} already exists in this folder.")

print("\nAll required cascades are ready!")