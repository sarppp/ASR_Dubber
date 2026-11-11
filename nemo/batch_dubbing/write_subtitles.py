"""Write requested subtitle sidecars from an existing supervised GPU result."""
import hashlib
import json
import sys
from pathlib import Path
from package_video import ROOT, srt


def main(index):
    work = ROOT / 'modal_control/batch_inputs'
    meta = json.loads((work / f'video_{index:02d}.json').read_text())
    result = json.loads((work / f'video_{index:02d}.gpu_full.json').read_text())
    if result['status'] != 'complete':
        raise ValueError('GPU result is incomplete')
    original = ROOT / meta['source']
    digest = hashlib.sha256()
    with original.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != meta['source_sha256']:
        raise ValueError('Original video changed since input extraction')
    duration = result['duration_seconds']
    rows = {'en': result['english'], **result['translations']}
    contents = {lang: srt(rows[lang], duration) for lang in ('en', 'de', 'fr')}
    outputs = {lang: original.with_name(original.stem + f'.{lang}.srt') for lang in contents}
    for lang, path in outputs.items():
        if path.exists() and path.read_bytes() != contents[lang]:
            raise FileExistsError(f'Conflicting subtitle: {path}')
    for lang, path in outputs.items():
        if not path.exists():
            with path.open('xb') as f:
                f.write(contents[lang])
    print(json.dumps({lang: str(path) for lang, path in outputs.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main(int(sys.argv[1]))
