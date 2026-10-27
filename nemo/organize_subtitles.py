#!/usr/bin/env python3
"""Move completed videos and sidecar subtitles into one folder per video.

Watch a running batch without moving its active or pending inputs:
  python3 organize_subtitles.py --input-dir new_ones --watch
"""

import argparse
import json
import time
from pathlib import Path


def organize_video(video: Path) -> Path:
    folder = video.parent / video.stem
    if not video.exists():
        relocated = folder / video.name
        if relocated.is_file():
            return relocated
        raise FileNotFoundError(video)
    folder.mkdir(exist_ok=True)
    sidecars = list(video.parent.glob(f"{video.stem}.*.srt"))
    direct_srt = video.with_suffix(".srt")
    if direct_srt.exists():
        sidecars.append(direct_srt)
    files = [video, *sidecars]
    # Check every destination before moving any file; never overwrite user data.
    for source in files:
        destination = folder / source.name
        if destination.exists():
            raise FileExistsError(destination)
    for source in files:
        source.rename(folder / source.name)
    return folder / video.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("new_ones"))
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    root = args.input_dir.resolve()
    report_path = root / "subtitle_local_report.json"
    organization_path = root / "subtitle_folders.json"
    relocated = {}
    if organization_path.exists():
        relocated = json.loads(organization_path.read_text())
    while True:
        report = json.loads(report_path.read_text())
        jobs = report["jobs"]
        for filename in dict.fromkeys(job["video"] for job in jobs):
            video = Path(filename)
            if video.parent != root:
                continue
            related = [job for job in jobs if job["video"] == filename]
            if not all(job["status"] in ("complete", "existing") for job in related):
                continue
            if filename not in relocated:
                destination = organize_video(video)
                relocated[filename] = str(destination)
                organization_path.write_text(json.dumps(relocated, indent=2) + "\n")
                print(f"Organized: {destination.parent.name}", flush=True)
        finished = all(job["status"] in ("complete", "existing") for job in jobs)
        if finished:
            # The worker has completed its last report write. Update its paths
            # only now, so it cannot overwrite them with old in-memory paths.
            for job in jobs:
                if job["video"] in relocated:
                    video = Path(relocated[job["video"]])
                    job["video"] = str(video)
                    job["output"] = str(video.parent / Path(job["output"]).name)
            temporary = report_path.with_suffix(".tmp.json")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(report_path)
            print("All completed videos and subtitles organized.", flush=True)
            return
        if report.get("status") == "failed":
            raise RuntimeError(f"Batch stopped: {report.get('error')}")
        if not args.watch:
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
