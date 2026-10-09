"""Install exact known Release assets after verifying uncompressed SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile
import os
ROOT = Path(__file__).resolve().parents[1]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-dir', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((ROOT / 'release_assets_manifest.json').read_text(encoding='utf-8'))
    for row in rows:
        if row['distribution'] != 'separate_release_asset':
            continue
        target = (ROOT / row['path']).resolve()
        target.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            h = hashlib.sha256()
            with target.open('rb') as f:
                for b in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(b)
            if h.hexdigest() != row['sha256']:
                raise ValueError('Refusing to replace changed asset: ' + row['path'])
            print('Already verified: ' + row['path'])
            continue
        with zipfile.ZipFile(args.asset_dir / row['asset_name']) as z:
            if z.namelist() != [row['path']] or z.getinfo(row['path']).file_size != row['size_bytes']:
                raise ValueError('Unexpected archive layout: ' + row['asset_name'])
            temp = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as f, z.open(row['path']) as source:
                    temp = Path(f.name)
                    h = hashlib.sha256()
                    for b in iter(lambda: source.read(1024 * 1024), b''):
                        f.write(b)
                        h.update(b)
                if h.hexdigest() != row['sha256'] or temp.stat().st_size != row['size_bytes']:
                    raise ValueError('Asset checksum failed: ' + row['path'])
                os.replace(temp, target)
                print('Installed: ' + row['path'])
            finally:
                if temp is not None and temp.exists():
                    temp.unlink()
if __name__ == '__main__':
    main()
