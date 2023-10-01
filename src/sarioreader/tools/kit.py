import urllib

import tqdm


def download_with_progressbar(url, filename):
    response = urllib.request.urlopen(url)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    t = tqdm.tqdm(total=total_size, unit="B", unit_scale=True, desc=filename)

    with open(filename, "wb") as f:
        while True:
            chunk = response.read(block_size)
            if not chunk:
                break
            t.update(len(chunk))
            f.write(chunk)
    t.close()
