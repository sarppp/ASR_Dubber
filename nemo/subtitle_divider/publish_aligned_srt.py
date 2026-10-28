"""Validate a completed translation and publish only SRT, with recoverable backups."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from aligned_translation import MODEL, REVISION, validate_complete
from bilingual_subtitles import write_subtitles


def publish(source_file: Path, result_file: Path, *, apply: bool = False) -> dict:
    source = json.loads(source_file.read_text(encoding="utf-8"))
    result = json.loads(result_file.read_text(encoding="utf-8"))
    original = Path(source["source"])
    if (result.get("status") != "complete" or result.get("source_sha256") != source["source_sha256"]
            or result.get("model") != MODEL or result.get("revision") != REVISION):
        raise ValueError("Result is incomplete or its source/model provenance differs")
    if hashlib.sha256(original.read_bytes()).hexdigest() != source["source_sha256"]:
        raise ValueError("English source changed after translation was prepared")
    if not original.name.endswith(".en.srt"):
        raise ValueError("Expected the original English .en.srt source")
    cues = validate_complete(result["rows"], source["cues"])
    folder = original.parent
    stem = original.name[:-len(".en.srt")]
    destinations = [folder / f"{stem}.srt", folder / f"{stem}.en-de-fr.srt",
                    folder / f"{folder.name}.en-de-fr.fixed.srt"]
    before = {path: path.read_bytes() if path.exists() else None for path in destinations}
    with tempfile.TemporaryDirectory(prefix="srt-validation-") as temporary:
        staged = Path(temporary) / "validated.srt"
        write_subtitles(staged, cues)
        data = staged.read_bytes()
    receipt = {"status": "preview", "cue_count": len(cues),
               "outputs": [str(p) for p in destinations],
               "sha256": hashlib.sha256(data).hexdigest()}
    if not apply:
        return receipt
    backup = Path(tempfile.mkdtemp(prefix="srt-before-repair-", dir=folder))
    for index, path in enumerate(destinations):
        if before[path] is not None:
            (backup / f"{index}.srt").write_bytes(before[path])
    (backup / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    replaced = []
    try:
        for path in destinations:
            current = path.read_bytes() if path.exists() else None
            if current != before[path]:
                raise ValueError(f"Subtitle changed during publication: {path}")
            fd, temp_name = tempfile.mkstemp(prefix=".srt-stage-", dir=folder)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp_name, path)
                replaced.append(path)
            finally:
                Path(temp_name).unlink(missing_ok=True)
    except Exception:
        for path in reversed(replaced):
            if before[path] is None:
                path.unlink()
            else:
                path.write_bytes(before[path])
        raise
    return {**receipt, "status": "published", "backup": str(backup)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(publish(args.source, args.result, apply=args.apply), indent=2))
