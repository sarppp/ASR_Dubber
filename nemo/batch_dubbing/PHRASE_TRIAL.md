# Ten-minute CRISPR experiment

This is an isolated experimental pipeline, not a replacement for the default
dubbing pipeline. Its input is the first 600 seconds of the CRISPR interview;
its artifacts live under
`experiments/crispr_phrase_modal_2026-10-30_10min_qwen/` and stay out of Git.

All inference runs on Modal, serially, with one T4 container at most:

1. Fresh Parakeet ASR and NeMo speaker analysis, without Canary translation.
2. Qwen3-14B-AWQ translates one source-linked English phrase into German and
   French per request. Adjacent English phrases provide context. All three
   subtitle languages share ASR word boundaries. A rejected overlong phrase
   can split automatically at a source word boundary; no source words are lost.
3. Qwen3-TTS-1.7B-CustomVoice speaks complete German utterances assembled from
   the translated phrases. Stock voices are not identity cloning. Native takes
   are retained. Timing conflicts block publication rather than clipping speech.
4. Local layout validation writes SRTs and ASS previews, then stream-copies the
   original video with the German audio into a separate ten-minute MP4.

Workers pin their model revisions and dependencies. Translation uses the vLLM
0.9.2 V0 engine with XGrammar. Exact phrase count, IDs, text length and rendered
width are validated separately. The earlier LM Format Enforcer outputs were
rejected after accented characters were corrupted; those checkpoints are audit
evidence, not approved target text.

`prepare_phrase_trial.py` creates the bounded controller catalog and batch from
the existing hashed `input/source.wav` and `input/manifest.json`. It never
launches inference. Inspect `./modal-control batch-plan <trial>/batch.json` and
use the controller's release workflow. Preparation does not authorize execution.
Do not edit pinned sources while a released batch is running. Outputs are
exclusive-create; a retry must preserve earlier evidence and declare its inputs.

`phrase_trial_report.py` reads provider billing and controller timing records.
The controller captures billing before execution, periodically, and after
shutdown. Posted credits can lag or be revised. Workspace credit changes may
include unrelated usage; app-specific charges and phase outcomes are reported
separately. Total experiment cost includes failed attempts, smoke tests and
setup. It must not be described as a measured clean repeat-run cost.

Structural checks do not prove translation quality or natural-sounding dubbing.
The artifacts explicitly retain `semantic_quality_verified=False` and
`listening_review_complete=False` until a separate review is performed.

Local tests load no inference models and launch no cloud jobs:

```sh
python -m pytest -q batch_dubbing/test_phrase_trial.py batch_dubbing/test_speech_stage.py
```
