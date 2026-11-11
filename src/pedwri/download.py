import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

ARTICLE = 14825193
API = "https://api.figshare.com/v2/articles/{}/files?page_size=100"


def main(argv=None):
    p = argparse.ArgumentParser(description="Download and extract GRAZPEDWRI-DX from figshare")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--keep-zips", action="store_true")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(API.format(ARTICLE)) as r:
        files = json.load(r)
    for f in files:
        dest = a.out / f["name"]
        print(f"{f['name']} ({f['size'] / 1e9:.2f} GB)")
        if not dest.exists():
            req = urllib.request.Request(f["download_url"], headers={"User-Agent": "pedwri"})
            with urllib.request.urlopen(req) as r, open(dest, "wb") as fh:
                shutil.copyfileobj(r, fh, length=1 << 20)
        if zipfile.is_zipfile(dest):
            with zipfile.ZipFile(dest) as z:
                z.extractall(a.out)
            if not a.keep_zips:
                dest.unlink()


if __name__ == "__main__":
    main()
