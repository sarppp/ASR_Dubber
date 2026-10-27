"""Build a complete evidence table from saved attempts; never invent missing results."""

import csv, html, json, re
from pathlib import Path
from modal_control.core import Store, utc

ROOT = Path(__file__).resolve().parents[1] / "speech_model_benchmark_2026-10-27"


def words(text):
    return re.findall(r"\w+", text.lower())


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = nxt
    return row[-1]


def main():
    models = json.loads((ROOT / "01_protocol/models.json").read_text())
    store = Store(ROOT.parent / "modal_control/.state")
    history = store.overview()["runs"]
    rows = []
    reference = ROOT / "runs/qwen_asr_17b/asr/transcript.txt"
    ref = words(reference.read_text()) if reference.exists() else []
    notes = {
        "whistle": "Frequent lexical errors in inspected passages (including book titles).",
        "parakeet_int8": "Strong CPU speed; inspected text has proper-name and book-title errors.",
        "qwen_asr_17b": "Generally coherent inspected text; Harari name still wrong. Agreement anchor only, not verified reference.",
        "qwen_asr_gguf_q8": "CLI reloads the model for each 25-second chunk; timing includes that overhead.",
        "granite_5": "German draft has awkward wording and a major meaning reversal in the nationalism/love passage; review required.",
        "ming_05b": "Initial German run omitted much of the supplied text. Configuration investigation in progress.",
        "supertonic_3": "German round-trip intelligibility is strong on this script. Preset voice; no speaker cloning.",
        "canary_current": "CURRENT baseline. Includes alignment. German preserves the love/hate negation that Granite reversed, but loses 99.99% precision, repeats one clause and leaves an English fragment.",
        "parakeet_current": "CURRENT baseline. Native GPU model with timestamps.",
    }
    for model in models:
        key = model["key"]
        run = ROOT / "runs" / key
        hardware = (
            json.loads((run / "hardware.json").read_text()).get("gpu")
            if (run / "hardware.json").exists()
            else None
        )
        gpu_name = (
            hardware.splitlines()[1].split(",")[0]
            if hardware and len(hardware.splitlines()) > 1
            else "GPU pending"
        )
        latest = next((x for x in history if x["job"] == key), None)
        timing_path=run/"client_timing.json"
        remote_wall=json.loads(timing_path.read_text()).get("remote_call_seconds") if timing_path.exists() else None
        resources = []
        if latest:
            with store.db() as c:
                resources = [
                    json.loads(r["payload"])
                    for r in c.execute(
                        "SELECT payload FROM events WHERE run_id=? AND kind='worker_event'",
                        (latest["id"],),
                    )
                ]
        gpu = []
        for sample in resources:
            try:
                gpu.append(float(sample["gpu_csv"].split(",")[1]) / 1024)
            except (KeyError, ValueError):
                pass
        for field in model["fields"]:
            dest = run / field
            metrics = (
                json.loads((dest / "metrics.json").read_text())
                if (dest / "metrics.json").exists()
                else {}
            )
            proxy = (
                json.loads((dest / "quality_proxy.json").read_text())
                if (dest / "quality_proxy.json").exists()
                else {}
            )
            wrong_gpu = model["hardware"] == "gpu" and (
                not hardware or "T4" not in hardware
            )
            valid = metrics.get("status") == "complete" and not wrong_gpu
            failure = ""
            for p in [
                run / "failure.json",
                dest / "failure.json",
                ROOT / "smoke_runs" / key / "failure.json",
                ROOT / "smoke_runs" / key / field / "failure.json",
            ]:
                if p.exists():
                    failure = json.loads(p.read_text()).get("error", "")
            if (
                latest
                and latest["status"]
                in ("preflight_failed", "failed", "startup_timeout")
                and not failure
            ):
                with store.db() as c:
                    errors = c.execute(
                        "SELECT payload FROM events WHERE run_id=? AND kind IN ('preflight_rejected','alert') ORDER BY seq DESC LIMIT 5",
                        (latest["id"],),
                    ).fetchall()
                failure = "; ".join(
                    json.loads(e["payload"]).get(
                        "reason", json.loads(e["payload"]).get("message", "")
                    )
                    for e in errors
                )
            agreement = None
            if (
                valid
                and field == "asr"
                and key != "qwen_asr_17b"
                and ref
                and (dest / "transcript.txt").exists()
            ):
                agreement = (
                    100
                    * distance(ref, words((dest / "transcript.txt").read_text()))
                    / len(ref)
                )
            status = (
                "complete"
                if valid
                else metrics.get("status")
                or (
                    latest["status"] if latest else ("failed" if failure else "pending")
                )
            )
            if wrong_gpu:
                proxy = {}
                status = latest["status"] if latest and latest["status"] in ("running","waiting","preflight") else "not_completed_on_T4"
                failure = (failure+"; " if failure else "")+"Non-T4 results excluded. Further tests stopped at user request."
            rows.append(
                {
                    "model": key,
                    "baseline": "CURRENT"
                    if key in ("canary_current", "parakeet_current")
                    else "candidate",
                    "field": field,
                    "hardware": gpu_name + " / 4 CPU"
                    if model["hardware"] == "gpu"
                    else "4 CPU",
                    "status": status,
                    "seconds": round(metrics["inference_seconds"], 2)
                    if valid
                    else None,
                    "speed_x": round(metrics["realtime_multiple"], 2)
                    if valid
                    else None,
                    "remote_call_wall_seconds":round(remote_wall,2) if valid and remote_wall is not None else None,
                    "warmup_seconds": round(metrics.get("warmup_seconds", 0), 2)
                    if valid
                    else None,
                    "load_seconds": round(metrics.get("load_seconds", 0), 2)
                    if valid
                    else None,
                    "peak_observed_GPU_GiB": round(max(gpu), 2) if gpu else None,
                    "ASR_disagreement_vs_Qwen17_percent": round(agreement, 2)
                    if agreement is not None
                    else None,
                    "TTS_roundtrip_WER_percent": round(proxy["wer"] * 100, 2)
                    if "wer" in proxy
                    else None,
                    "notes": (notes.get(key, "") if not wrong_gpu else "")
                    + ((" Failure: " + failure[:350]) if not valid and failure else ""),
                    "artifact_folder": str(dest.relative_to(ROOT)),
                    "model_repo": model["repo"],
                    "revision": model.get("revision"),
                }
            )
    out = ROOT / "05_evaluation"
    out.mkdir(exist_ok=True)
    (out / "full_comparison.json").write_text(
        json.dumps({"updated": utc(), "rows": rows}, indent=2)
    )
    with (out / "full_comparison.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    intro = """<h1>Current models vs candidates</h1><p>Same Harari / Ezra Klein 00:00–10:00 excerpt. GPU constraint: T4 ONLY, no fallback. Earlier L4/A10G measurements are excluded. CURRENT marks the existing Canary and Parakeet models. Each model competes within its own field. CPU and GPU numbers are separate configurations.</p><p>Speech recognition/translation speed = 600 source seconds divided by measured processing time. TTS speed = generated audio seconds divided by synthesis time; it does not mean the full source video was translated that quickly. Downloads, model loading and warm-up are separate. This is one measured run per configuration, not a statistically stable latency distribution.</p><p>ASR disagreement is normalized word edit distance against an earlier Qwen 1.7B transcript (generated on L4; used only as an unverified text reference, not a speed result), <b>not accuracy or ground truth</b>. That model can be wrong too. TTS WER is an independent Parakeet round-trip recognizer versus the supplied German script, <b>not human-rated naturalness or voice similarity</b>. Blank cells mean unmeasured. Peak GPU memory is sampled, so brief peaks can be missed. Translation quality notes are qualitative inspection, not a COMET/BLEU score.</p><p>Dubbing files use a single reference speaker (Supertonic uses preset M1), not speaker diarization or lip synchronization. The shared German script is assistant-reviewed, not a certified human translation. Review raw samples as well as the fitted video.</p>"""
    page = [
        '<!doctype html><meta charset="utf-8"><title>Full model comparison</title><style>body{font:14px system-ui;margin:30px;background:#101827;color:#e8eef8}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #35445c;text-align:left;vertical-align:top}th{position:sticky;top:0;background:#243148}a{color:#9bceff}p{max-width:1200px}td:last-child{min-width:260px}</style>',
        intro,
        "<p>Updated "
        + utc()
        + ' · <a href="full_comparison.csv">Full CSV with timings, revisions and paths</a> · <a href="../START_HERE.html">Play samples and open videos</a></p>',
    ]
    for field in ("asr", "translation", "tts", "speech_translation"):
        page.append(
            "<h2>"
            + field
            + "</h2><table><tr>"
            + "".join(
                "<th>" + x + "</th>"
                for x in [
                    "Model",
                    "Role",
                    "Hardware",
                    "State",
                    "Seconds",
                    "Speed ×",
                    "GPU GiB",
                    "ASR disagreement %",
                    "TTS WER proxy %",
                    "Assessment / limitation",
                ]
            )
            + "</tr>"
        )
        for r in rows:
            if r["field"] != field:
                continue
            keys = [
                "model",
                "baseline",
                "hardware",
                "status",
                "seconds",
                "speed_x",
                "peak_observed_GPU_GiB",
                "ASR_disagreement_vs_Qwen17_percent",
                "TTS_roundtrip_WER_percent",
                "notes",
            ]
            page.append(
                "<tr>"
                + "".join(
                    "<td>"
                    + html.escape(str(r[k]) if r[k] is not None else "—")
                    + "</td>"
                    for k in keys
                )
                + "</tr>"
            )
        page.append("</table>")
    (out / "ASSESSMENT.html").write_text("".join(page))
    print(
        "Comparison updated:",
        len(rows),
        "field entries,",
        sum(r["status"] == "complete" for r in rows),
        "complete",
    )


if __name__ == "__main__":
    main()
