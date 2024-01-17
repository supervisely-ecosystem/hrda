import re
import sys, os
import shutil
import requests


def download_file(url, path):
    r = requests.get(url, stream=True)
    mimetype = r.headers['content-type']
    ext = mimetype.split('/')[-1]
    filename = path + '.' + ext
    with open(filename, 'wb') as f:
        r.raw.decode_content = True
        shutil.copyfileobj(r.raw, f)
    return filename


def read_ipynb(path):
    with open(path) as f:
        nb = f.read()
    return nb


def write_ipynb(nb, path):
    with open(path, 'w') as f:
        f.write(nb)


def find_images(nb):
    return re.findall(r'\!\[.*?\]\(.*?\)', nb)


if __name__ == '__main__':
    os.makedirs('assets', exist_ok=True)
    nb = read_ipynb('inference_outside_supervisely.ipynb')
    image_paths = find_images(nb)

    print(f"Found {len(image_paths)} images")
    print(image_paths)
    print("Only first two images will be replaced")

    for entry in image_paths[:2]:
        full_text = entry
        alt_text = entry.split('[')[-1].split(']')[0]
        url = entry.split('(')[-1].split(')')[0]

        # alt_text to alphanumerics
        filename = re.sub(r' ', '_', alt_text)
        filename = re.sub(r'[^a-zA-Z0-9_]', '', filename)

        print('Downloading', url)
        save_path = download_file(url, 'assets/' + filename)
        nb = nb.replace(url, save_path)

    write_ipynb(nb, 'inference_outside_supervisely.ipynb')