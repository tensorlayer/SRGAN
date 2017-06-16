import os
import urllib
import numpy as np

from joblib import Parallel, delayed


def download_image(download_str, save_dir):
    img_name, img_url = download_str.strip().split('\t')
    save_img = os.path.join(save_dir, "{}.jpg".format(img_name))
    try:
        if not os.path.isfile(save_img):
            print("Downloading {} to {}.jpg".format(img_url, img_name))
            urllib.urlretrieve(img_url, save_img)
        else:
            print("Already downloaded {}".format(save_img))
    except Exception:
        print("File not exists.")


def main():
    np.random.seed(123456)
    url_file = "/data/imagenet/fall11_urls.txt"
    save_dir = "/data/imagenet/"
    n_download_imgs = 20000

    with open(url_file) as f:
        lines = f.readlines()
        lines = np.random.choice(lines, size=n_download_imgs, replace=False)

    Parallel(n_jobs=12)(delayed(download_image)(line, save_dir) for line in lines)


if __name__ == "__main__":
    main()
