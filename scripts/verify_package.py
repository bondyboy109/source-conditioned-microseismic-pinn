"""Check current distribution integrity; historical protocol differences are separate."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-assets', action='store_true')
    args = parser.parse_args()
    rows = json.loads((ROOT / 'SHA256_MANIFEST.json').read_text(encoding='utf-8'))['files']
    problems = []
    for row in rows:
        p = ROOT / row['path']
        if not p.is_file() or p.stat().st_size != row['size_bytes'] or digest(p) != row['sha256']:
            problems.append(row['path'])
    assets = json.loads((ROOT / 'release_assets_manifest.json').read_text(encoding='utf-8'))
    for row in assets:
        p = ROOT / row['path']
        required = args.require_assets and row['distribution'] == 'separate_release_asset'
        if p.is_file():
            if p.stat().st_size != row['size_bytes'] or digest(p) != row['sha256']:
                problems.append(row['path'])
        elif required:
            problems.append(row['path'])
    if problems:
        raise SystemExit('Integrity check failed: ' + ', '.join(problems))
    print(f'Verified {len(rows)} repository files; historical provenance limitations remain documented.')
if __name__ == '__main__':
    main()
