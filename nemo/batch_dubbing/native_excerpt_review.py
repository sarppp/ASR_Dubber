"""Build an offline listening review from existing files; no media encoding."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
from urllib.parse import quote


def write_review(input_file, output_file, report_file=None):
    root = Path(__file__).resolve().parents[1]
    data = json.loads(Path(input_file).read_text())
    output = Path(output_file).resolve()
    source = data['source_metadata']
    escape = html.escape

    def url(path):
        return escape(quote(os.path.relpath(Path(path).resolve(), output.parent), safe='/'))

    videos = []
    for key, title in [('original_video', 'Original English'), ('older_dub', 'Existing German dub')]:
        path = root / source[key]
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != source[key + '_sha256']:
            raise ValueError('Reference video changed')
        videos.append(f'<article><h2>{title}</h2><video controls preload="metadata" '
                      f'src="{url(path)}#t={data["excerpt_start"]},{data["excerpt_end"]}" '
                      f'data-start="{data["excerpt_start"]}" data-end="{data["excerpt_end"]}"></video></article>')
    audio, diagnostics = [], '<p>New audio has not been generated yet.</p>'
    if report_file is not None:
        report_path = Path(report_file).resolve()
        report = json.loads(report_path.read_text())
        if report.get('status') != 'complete' or report.get('stage') != 'workload' or report.get('input') != data:
            raise ValueError('A successful matching workload report is required')
        for language in ('en', 'de'):
            row = next(x for x in report['sequences'] if x['file'] == f'native_{language}.wav')
            path = report_path.parent / row['file']
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Listening sequence changed')
            title = 'New English' if language == 'en' else 'New German'
            audio.append(f'<article><h2>{title} · 1.0×</h2><audio controls preload="metadata" '
                         f'src="{url(path)}"></audio><p>{row["duration_seconds"]:.2f} seconds. '
                         'Full native audio with caption-derived gaps; its timing may differ from the video.</p></article>')
        rows = report['strict_caption_placement']['rows']
        cells = ''.join(f'<tr><td>{escape(r["id"])}</td><td>{r["target_frames"]/44100:.3f}</td>'
                        f'<td>{r["native_frames"]/44100:.3f}</td>'
                        f'<td>{(r["native_frames"]-r["target_frames"])/44100:+.3f}</td></tr>' for r in rows)
        diagnostics = '<table><tr><th>Block</th><th>Caption window (s)</th><th>Raw audio (s)</th><th>Difference (s)</th></tr>' + cells + '</table>'
    texts = ''.join(f'<details><summary>{escape(row["id"])} · {row["srt_start"]:.2f}–{row["srt_end"]:.2f}s</summary>'
                    f'<p lang="{row["language"]}">{escape(row["text"])}</p></details>' for row in data['samples'])
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Native speech comparison</title><style>
body{font:17px/1.5 system-ui;background:#f5f4f0;color:#202c32;max-width:1100px;margin:auto;padding:24px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}article{background:white;border-radius:12px;padding:18px}
h1{font-size:1.7rem}h2{font-size:1.15rem}video,audio{width:100%}td,th{text-align:left;padding:8px;border-bottom:1px solid #ccc}details{margin:12px 0}
</style><h1>''' + escape(source.get('title', 'Native speech comparison')) + '''</h1>
<p>Compare the source, existing dub and complete new speech at normal playback speed. Only one player runs at a time.
Only gaps between grouped blocks and at the excerpt edges are inserted from captions. Within each block, the synthesizer controls pauses. Exact original speech and silence matching has not been verified.</p><div class="grid">''' + ''.join(videos + audio) + '''</div>
<h2>Timing check</h2>''' + diagnostics + '<h2>Words used for synthesis</h2>' + texts + '''
<script>const players=[...document.querySelectorAll('video,audio')];
for(const p of players){p.addEventListener('play',()=>{for(const q of players)if(q!==p)q.pause();
if(p.dataset.end&&(p.currentTime>=+p.dataset.end||p.currentTime<+p.dataset.start))p.currentTime=+p.dataset.start;});
p.addEventListener('timeupdate',()=>{if(p.dataset.end&&p.currentTime>=+p.dataset.end)p.pause();});}
</script></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-file', type=Path, required=True)
    parser.add_argument('--output-file', type=Path, required=True)
    parser.add_argument('--report-file', type=Path)
    args = parser.parse_args()
    write_review(args.input_file, args.output_file, args.report_file)
