import argparse
import socket
import os
import urllib
import numpy as np
from PIL import Image

from joblib import Parallel, delayed


def download_image(download_str, save_dir):
    img_name, img_url = download_str.strip().split('\t')
    save_img = os.path.join(save_dir, "{}.jpg".format(img_name))
    downloaded = False
    try:
        if not os.path.isfile(save_img):
            print("Downloading {} to {}.jpg".format(img_url, img_name))
            urllib.urlretrieve(img_url, save_img)

            # Check size of the images
            downloaded = True
            with Image.open(save_img) as img:
                width, height = img.size

            img_size_bytes = os.path.getsize(save_img)
            img_size_KB = img_size_bytes / 1024

            if width < 500 or height < 500 or img_size_KB < 200:
                os.remove(save_img)
                print("Remove downloaded images (w:{}, h:{}, s:{}KB)".format(width, height, img_size_KB))
        else:
            print("Already downloaded {}".format(save_img))
    except Exception:
        if not downloaded:
            print("Cannot download.")
        else:
            print("Remove failed, downloaded images.")

        if os.path.isfile(save_img):
            os.remove(save_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_url_file", type=str, required=True,
                       help="File that contains list of image IDs and urls.")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory where to save outputs.")
    parser.add_argument("--n_download_urls", type=int, default=20000,
                       help="Directory where to save outputs.")
    args = parser.parse_args()

    # np.random.seed(123456)

    socket.setdefaulttimeout(10)

    with open(args.img_url_file) as f:
        lines = f.readlines()
        lines = np.random.choice(lines, size=args.n_download_urls, replace=False)

    Parallel(n_jobs=12)(delayed(download_image)(line, args.output_dir) for line in lines)


if __name__ == "__main__":
    main()
