from pathlib import Path
import json, shutil, subprocess, html, csv, hashlib

ROOT = Path(__file__).resolve().parents[1] / "speech_model_benchmark_2026-10-27"
CATS = {
    "asr": "02_asr",
    "translation": "03_translation",
    "tts": "04_dubbing",
    "speech_translation": "04_dubbing",
}


def ff(args):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", *map(str, args)], check=True
    )


def stamp(t):
    m = round(t * 1000)
    return (
        f"{m // 3600000:02d}:{m // 60000 % 60:02d}:{m // 1000 % 60:02d},{m % 1000:03d}"
    )


def organize():
    rows = []
    for model in json.loads((ROOT / "01_protocol/models.json").read_text()):
        key = model["key"]
        run = ROOT / "runs" / key
        for field in model["fields"]:
            dest = ROOT / CATS[field] / key
            if field == "speech_translation":
                dest = dest / "direct_speech_translation"
            dest.mkdir(parents=True, exist_ok=True)
            src = run / field
            observed_hw = (
                json.loads((run / "hardware.json").read_text()).get("gpu")
                if (run / "hardware.json").exists()
                else None
            )
            gpu_name = (
                observed_hw.splitlines()[1].split(",")[0]
                if observed_hw and len(observed_hw.splitlines()) > 1
                else "GPU pending"
            )
            row = {
                "model": key,
                "field": field,
                "hardware": gpu_name + " + 4 CPU"
                if model["hardware"] == "gpu"
                else "4 CPU",
                "status": "pending",
                "seconds": None,
                "rtf": None,
                "folder": str(dest.relative_to(ROOT)),
            }
            for p in [
                run / "model.json",
                run / "hardware.json",
                run / "pip-freeze.txt",
                run / "download.json",
                run / "load.json",
            ]:
                if p.exists():
                    shutil.copy2(p, dest / p.name)
            if (run / "failure.json").exists():
                row["status"] = "failed"
                shutil.copy2(run / "failure.json", dest / "failure.json")
            if src.exists():
                for p in src.iterdir():
                    if p.is_file() and (
                        not (dest / p.name).exists()
                        or p.stat().st_mtime > (dest / p.name).stat().st_mtime
                    ):
                        shutil.copy2(p, dest / p.name)
                if (src / "failure.json").exists():
                    row["status"] = "failed"
                if (src / "metrics.json").exists():
                    metrics = json.loads((src / "metrics.json").read_text())
                    row.update(
                        status=metrics["status"],
                        seconds=metrics["inference_seconds"],
                        rtf=metrics["rtf"],
                    )
                if (src / "segments.json").exists():
                    segs = json.loads((src / "segments.json").read_text())
                    if all("text" in s for s in segs):
                        (dest / "captions_chunk_timing.srt").write_text(
                            "\n\n".join(
                                f"{i + 1}\n{stamp(s['source_start'])} --> {stamp(s['source_end'])}\n{s['text']}"
                                for i, s in enumerate(segs)
                            )
                            + "\n"
                        )
                    if (
                        field in ("tts", "speech_translation")
                        and row["status"] == "complete"
                        and (
                            not (dest / "dubbed_10min.mp4").exists()
                            or not (dest / "render_source.sha256").exists()
                            or (dest / "render_source.sha256").read_text()
                            != hashlib.sha256(
                                (src / "metrics.json").read_bytes()
                            ).hexdigest()
                        )
                    ):
                        with __import__("tempfile").TemporaryDirectory(
                            prefix="dub-fit-"
                        ) as tmp:
                            tmp = Path(tmp)
                            raw = []
                            fitted = []
                            for i, s in enumerate(segs):
                                p = dest / s["path"]
                                r = tmp / f"raw{i:03d}.wav"
                                f = tmp / f"fit{i:03d}.wav"
                                ff(["-i", p, "-ar", "24000", "-ac", "1", r])
                                raw.append(r)
                                duration = s["audio_seconds"]
                                window = s["source_end"] - s["source_start"]
                                ratio = max(1.0, duration / window)
                                factors = []
                                while ratio > 2:
                                    factors.append("atempo=2")
                                    ratio /= 2
                                if ratio > 1.001:
                                    factors.append(f"atempo={ratio:.8f}")
                                factors += ["apad", f"atrim=duration={window}"]
                                ff(["-i", r, "-af", ",".join(factors), f])
                                fitted.append(f)
                            for name, files in [
                                ("raw_narration.wav", raw),
                                ("timeline_600s.wav", fitted),
                            ]:
                                lst = tmp / (name + ".txt")
                                lst.write_text(
                                    "".join("file '" + str(p) + "'\n" for p in files)
                                )
                                ff(
                                    [
                                        "-f",
                                        "concat",
                                        "-safe",
                                        "0",
                                        "-i",
                                        lst,
                                        "-c",
                                        "copy",
                                        dest / name,
                                    ]
                                )
                            ff(
                                [
                                    "-i",
                                    ROOT / "00_inputs/source_10min.mp4",
                                    "-i",
                                    dest / "timeline_600s.wav",
                                    "-map",
                                    "0:v:0",
                                    "-map",
                                    "1:a:0",
                                    "-c:v",
                                    "copy",
                                    "-c:a",
                                    "aac",
                                    "-t",
                                    "600",
                                    "-movflags",
                                    "+faststart",
                                    dest / "dubbed_10min.mp4",
                                ]
                            )
                        (dest / "render_source.sha256").write_text(
                            hashlib.sha256(
                                (src / "metrics.json").read_bytes()
                            ).hexdigest()
                        )
                        for idx in (0, 8, 16):
                            shutil.copy2(
                                dest / segs[idx]["path"],
                                dest / f"review_{idx * 25:03d}s.wav",
                            )
                        (dest / "READ_ME.txt").write_text(
                            "raw_narration.wav: generated speech concatenated at its original speed.\ntimeline_600s.wav and dubbed_10min.mp4: each generated segment is sped up ONLY if needed to fit its 25-second source window, otherwise padded with silence. Original source audio is replaced. No lip synchronization or speaker-by-speaker voice matching is claimed.\nreview_000s.wav, review_200s.wav, review_400s.wav: the same three text passages across TTS entrants, at raw generated speed.\ncaptions_chunk_timing.srt uses the input chunk windows, not forced word alignment.\n"
                        )
            if model["hardware"] == "gpu" and "T4" not in gpu_name:
                row.update(
                    status="excluded_non_T4; further tests stopped", seconds=None, rtf=None
                )
            rows.append(row)
    (ROOT / "05_evaluation/status.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (ROOT / "05_evaluation/leaderboard.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    page = [
        '<!doctype html><meta charset="utf-8"><title>Speech model comparison — Harari, first 10 minutes</title><style>body{font:16px system-ui;background:#111827;color:#e5e7eb;max-width:1200px;margin:40px auto;padding:20px}a{color:#93c5fd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #374151;text-align:left}audio{width:230px}small{color:#9ca3af}h1{font-size:28px}</style><h1>Speech model comparison</h1><p>Harari / Ezra Klein · 00:00–10:00 · 600 seconds · German subtitles and speech.</p><p><a href="00_inputs/source_10min.mp4">Original 10-minute video</a> · <a href="00_inputs/german_script.txt">Shared German TTS script</a> · <a href="05_evaluation/leaderboard.csv">Measured timings CSV</a> · <a href="05_evaluation/ASSESSMENT.html">Assessment and caveats</a></p><p>Compare CPU and GPU separately. Inference excludes initial downloads and warm-up. GGUF CLI timings may include repeated process/model startup. Translation text is evaluated separately from speech synthesis. The common German script is assistant-reviewed, not human-verified ground truth.</p>'
    ]
    for field in ("asr", "translation", "tts", "speech_translation"):
        page.append(
            "<h2>"
            + html.escape(field)
            + "</h2><table><tr><th>Model / hardware</th><th>Status / speed</th><th>Files and matching review passage</th></tr>"
        )
        for r in [x for x in rows if x["field"] == field]:
            d = ROOT / r["folder"]
            links = []
            for name, label in [
                ("transcript.txt", "Transcript"),
                ("captions_chunk_timing.srt", "Chunk-timed captions"),
                ("dubbed_10min.mp4", "Dubbed video"),
                ("raw_narration.wav", "Raw narration"),
                ("metrics.json", "Metrics"),
                ("failure.json", "Error"),
            ]:
                if (d / name).exists():
                    links.append(f'<a href="{r["folder"]}/{name}">{label}</a>')
            if (d / "review_000s.wav").exists():
                links.append(
                    f'<br><audio controls preload="none" src="{r["folder"]}/review_000s.wav"></audio>'
                )
            speed = (
                ""
                if r["seconds"] is None
                else f"<br>{r['seconds']:.1f} s · {1 / r['rtf']:.1f}× real-time"
            )
            page.append(
                f"<tr><td>{r['model']}<br><small>{r['hardware']}</small></td><td>{r['status']}{speed}</td><td>"
                + " · ".join(links)
                + "</td></tr>"
            )
        page.append("</table>")
    (ROOT / "START_HERE.html").write_text("".join(page))
    print(
        "Updated",
        len(rows),
        "task entries;",
        sum(r["status"] == "complete" for r in rows),
        "complete",
    )


if __name__ == "__main__":
    organize()
