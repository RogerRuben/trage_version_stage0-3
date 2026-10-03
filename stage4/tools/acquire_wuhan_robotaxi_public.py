"""Download a bounded, pinned public research snapshot; never execute upstream code."""
from pathlib import Path
import hashlib
import json
import urllib.request
import time

REV = 'd8de2c56e6c127542c98c9d224021cd131cd4f77'
REPO = 'Tlab-seu-code/Urban_Robotaxi_Analysis-'
ROOT = Path(__file__).resolve().parents[1] / 'output/external_research/wuhan_robotaxi_2026'

def fetch(url, dest, expected=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'PublicResearchDataAcquisition/1.0'})
                with urllib.request.urlopen(request, timeout=90) as response, dest.with_suffix(dest.suffix+'.part').open('wb') as out:
                    while chunk := response.read(1024*1024):
                        out.write(chunk)
                dest.with_suffix(dest.suffix+'.part').replace(dest)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    with dest.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    if expected and digest != expected:
        raise ValueError(f'SHA mismatch: {dest}')
    print(f'{dest.relative_to(ROOT)}: {dest.stat().st_size}', flush=True)
    return {'path': str(dest.relative_to(ROOT)), 'url': url, 'bytes': dest.stat().st_size, 'sha256': digest}

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []
    api = f'https://api.github.com/repos/{REPO}'
    fetch(f'{api}/git/trees/{REV}?recursive=1', ROOT/'upstream_tree.json')
    fetch(f'{api}/releases/tags/v1.0', ROOT/'release_v1.json')
    tree = json.loads((ROOT/'upstream_tree.json').read_text(encoding='utf-8'))['tree']
    for entry in tree:
        path = entry['path']
        if entry['type'] != 'blob' or path.endswith(('.pkl', '.log')):
            continue
        if '/NYC-sample/' in path or '/Porto-sample/' in path:
            continue
        if 'carpooling_one_day_results_v12_sumo/' in path and not path.endswith(('sensitivity_summary.csv', 'sensitivity_summary.json')):
            continue
        if path == '7-sample_data_and_code/carpooling_one_day_sample_code/robust.net.xml':
            continue  # same public network; do not store a second 288 MB copy
        selected = (entry['size'] < 100000 and path.endswith(('.py','.md','.R','.txt','.xml','.ps1')))
        selected |= 'Wuhan-sample/' in path
        selected |= 'carpooling_one_day_sample_code/' in path and path.endswith(('.csv','.json'))
        if not selected:
            continue
        url = f'https://raw.githubusercontent.com/{REPO}/{REV}/{path}'
        dest = ROOT/'repository'/path
        item = fetch(url, dest)
        if dest.stat().st_size < 200 and dest.read_bytes().startswith(b'version https://git-lfs'):
            pointer = dest.read_text()
            sha = pointer.split('oid sha256:')[1].splitlines()[0]
            size = int(pointer.split('size ')[1].strip())
            if size > 400_000_000:
                continue
            pointer_dest = ROOT/'lfs_pointers'/path
            pointer_dest.parent.mkdir(parents=True, exist_ok=True)
            dest.replace(pointer_dest)
            try:
                item = fetch(f'https://media.githubusercontent.com/media/{REPO}/{REV}/{path}', dest, sha)
            except Exception as exc:
                manifest.append({'path': path, 'lfs_sha256': sha, 'bytes_expected': size, 'error': str(exc)})
                continue
            if item['bytes'] != size:
                raise ValueError(f'LFS size mismatch: {path}')
            item['lfs_verified'] = True
        # On resume, the local payload is no longer an LFS pointer; still verify it.
        saved_pointer = ROOT/'lfs_pointers'/path
        if saved_pointer.is_file():
            pointer = saved_pointer.read_text()
            expected_sha = pointer.split('oid sha256:')[1].splitlines()[0]
            expected_size = int(pointer.split('size ')[1].strip())
            if item['sha256'] != expected_sha or item['bytes'] != expected_size:
                raise ValueError(f'Resumed LFS verification failed: {path}')
            item['lfs_verified'] = True
        manifest.append(item)
        (ROOT/'download_manifest.json').write_text(json.dumps({'revision': REV, 'files': manifest}, indent=2), encoding='utf-8')
    base = 'https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41893-026-01944-2/MediaObjects/'
    for number in range(1, 13):
        name = f'41893_2026_1944_MOESM{number}_ESM.' + ('pdf' if number == 1 else 'xlsx')
        try:
            manifest.append(fetch(base+name, ROOT/'publisher'/name))
        except Exception as exc:
            manifest.append({'url':base+name, 'error':str(exc)})
    (ROOT/'download_manifest.json').write_text(json.dumps({'revision': REV, 'files': manifest}, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
