# English, French, and German subtitles

Run from this `nemo` directory. The runner processes media files inside
`new_ones` and its visible subfolders and saves `<video>.nemo.en.srt`, `.nemo.fr.srt`, and
`.nemo.de.srt` alongside it. It assumes the videos contain English audio.

## Local NVIDIA GPU

```bash
.venv-local-asr/bin/python -u batch_subtitles.py --backend local --input-dir new_ones
# English first; translations can be run separately afterward.
.venv-local-asr/bin/python -u batch_subtitles.py --backend local --input-dir new_ones --languages en
```

The local backend uses the cached `nvidia/canary-1b-v2` model for all three
languages. It runs one inference at a time in FP16, with chunks of at most
25 seconds and boundaries selected from pauses. Its auxiliary timestamp
alignment model is also converted to FP16. English uses word timestamps;
French and German use translated segment timestamps.
If Canary cannot align source-language speech, the local backend automatically
uses the loaded CTC model's transcription and word timestamps for that chunk.
It does not substitute source-language text for a failed translation.

The original `.venv` is preserved. The separate runtime uses NeMo 2.5.3 for
timestamped translations. To recreate it with Python 3.12:

```bash
uv venv --python 3.12 .venv-local-asr
uv pip install --python .venv-local-asr/bin/python \
  --index-strategy unsafe-best-match -r local-asr-requirements.txt
```

The model must already be in the Hugging Face cache; the local runner does
not upload the media or invoke Modal. GPU access is required.

Progress is recorded in `new_ones/subtitle_local_report.json`. The current
run logs to `new_ones/subtitle_local.log`. Intermediate audio and chunk
checkpoints are in `new_ones/.local_asr/`. The same command resumes completed
chunks and skips valid SRT files. Keep the cache until the batch is finished.
Run only one local batch at a time.

To organize outputs while a batch is running, use
`python3 organize_subtitles.py --input-dir new_ones --watch`. Each completed
video and its SRTs move into `new_ones/<video-stem>/`. Active and pending
inputs stay in place until their jobs finish. Future subtitle runs discover
the organized videos and save additional languages in the same folders.

Audio extraction preserves timestamp gaps in the source media so the
subtitles stay on the video's original timeline. SRT syntax, ordering, and
time bounds are checked before final files are saved. These checks do not
guarantee transcription or translation accuracy.

## Preview or Modal

```bash
python3 batch_subtitles.py --backend local --input-dir new_ones --dry-run
uv run --no-sync --env-file .env python batch_subtitles.py --input-dir new_ones --gpu T4
```

Modal uses Parakeet for English and Canary for French and German, with two
remote calls at a time by default. It requires GPU access on the configured
Modal workspace. The local backend does not use the Modal `--gpu` or `--jobs`
settings.
