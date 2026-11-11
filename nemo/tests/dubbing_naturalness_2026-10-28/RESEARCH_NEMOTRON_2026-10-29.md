# Nemotron and longer-sentence dubbing checks

Checked 2026-10-29. Research/preparation below; the subsequently released
synthesis completed. The first OpenRouter upload returned HTTP404 because all
providers were ignored by account settings. See HANDOFF.md for measured results.

## Selected route

OpenRouter lists `nvidia/nemotron-3.5-transcribe`, canonical release
`nvidia/nemotron-3.5-transcribe-20261028`, with a rate of $0.0000333 per audio
second: $0.001998/minute, $0.01998/ten minutes, or $0.005994 for the proposed
180-second maximum. The current endpoint is DeepInfra BF16. This uses OpenRouter
billing; the user does not need a separate DeepInfra account. The endpoint lists
speaker diarization as unsupported. These are catalog observations, not a
successful inference benchmark. [Official transcription catalog](https://openrouter.ai/api/v1/models?output_modalities=transcription),
[endpoint metadata](https://openrouter.ai/api/v1/models/nvidia/nemotron-3.5-transcribe/endpoints).

The API is `/api/v1/audio/transcriptions`, with JSON base64 audio, explicit
language, `verbose_json`, and requested word/segment timestamps. The response can
include `usage.cost`; `X-Generation-Id` identifies the request. Structured fields
vary by backend. Ordinary provider `only`/`order` settings do not constrain this
transcription route. No retry or model fallback is planned. Missing timestamps
must remain unverified, never be synthesized from text. [Official API guide](https://openrouter.ai/docs/guides/overview/multimodal/stt).

User selected OpenRouter after declining DeepInfra billing. Credential discovery
was explicitly authorized in Budget_App. Its root `.env` contains `OPEN_ROUTER`;
keep the secret there and load it only into memory for the released request.
No credential values belong in this repository or generated reports.

## What the new model can establish

NVIDIA released the model on October28. It processes speech into text, including
English/German transcription and translation; it does not synthesize dubbed
speech. About34B total parameters and4.3B active parameters are different memory
quantities. The BF16 quickstart uses H10080GB. This is not a6GB local deployment
candidate. [NVIDIA model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B).

The technical report describes a CTC aligner with40ms frames. Reported German
word-boundary F1 is87.12% on its MCS evaluation, using a±200ms tolerance. Those
numbers are NVIDIA's evaluation, not our videos, and do not demonstrate exact
speech/silence boundaries. Its transcript can help inspect sentence endings,
while acoustic timing and listening remain independent checks.
[NVIDIA technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Transcribe-Technical-Report.pdf).

For our comparison, keep expected text local: do not feed it as a recognition
hint. Record differences and terminal-word coverage, but label them potential
ASR or synthesis errors rather than automatically blaming the synthesizer.
Correct words alone cannot establish naturalness or source-pause agreement.

## Why not download first

The official C++ path supports CPU and older GPUs through upstream llama.cpp
with IQ4_NL; it needs matching decoder and encoder GGUF files. The optimized
batched GPU implementation requires compute capability8.0+, whereas T4 is7.5.
[Official platform documentation](https://raw.githubusercontent.com/NVIDIA/NeMo-Speech.cpp/main/docs/salm/models.md),
[NVIDIA GPU capabilities](https://developer.nvidia.com/cuda/gpus).

The published smaller GPU configuration uses24GB for IQ4_NL. This does not prove
that a serial short request cannot fit a smaller configuration, but neither T4
memory fit nor useful CPU latency has been measured here. Decoder/encoder GGUF
download sizes could not be verified because linked repository pages returned401.
No weight files were downloaded. A successful cold start below$0.10 is therefore
unestablished. OpenRouter avoids this setup for the current test.
[Official memory configuration](https://raw.githubusercontent.com/NVIDIA/NeMo-Speech.cpp/main/docs/salm/configuration.md),
[official checkpoint links](https://research.nvidia.com/labs/nemotron/NVIDIA-Nemotron-3.5-Transcribe/).

## Longer sentence test

The synthetic fixture uses six API calls: short/long/short in English, then
short/long/short in German. Each long passage has73words and a short final
sentence. Cached Supertonic1.3.1 splits the six texts into1,2,1,1,2,1chunks,
verified by extracting its pure chunker without loading the model. Its300-character
setting does not hard-split a single long sentence: the first long sentence
exceeds300characters and remains intact. Never estimate calls using character
count divided by300.

The worker uses explicit1.0speed, eightsteps, M1, fixed per-sample seeds, and the
existing cached image/revision. Smoke produces only the first English short;
workload reuses it and synthesizes the other five. Every raw sample stays intact.
Separate language sequences insert0.4seconds of zero samples between clips;
Supertonic's internal chunk pause is explicitly0.3seconds. Native edge silence
is additional, so these insertions are not claims about total perceived silence.

Placement verification must compare sample hashes, counts and boundaries. A
deadline shorter than the complete clip or a collision must block placement,
not cut audio, accelerate it, or call the result fixed. This test exercises
generation and joins; it does not yet align English and German to the same
original source intervals. The production integration still needs a suitable
duration/pause-conditioned synthesizer or bounded faithful wording candidates.

Planned resources: one Modal CPU container at a time,2CPU/4GiB, noGPU, no build,
no download, no retries. Each stage has150s function timeout,30s startup and180s
controller deadline; two stages maximum. At reviewed CPU/RAM rates,360s is
$0.0126288 before transfer/cleanup. With24s allowance and128MiB conservative
transfer at$0.04/GiB, estimate$0.01848. Add the180s OpenRouter allowance and total
is about$0.02448, below the requested$0.10 target. These are estimates, not a
provider-enforced currency ceiling. No automatic spending increase after failure.
[Modal pricing](https://modal.com/pricing).

Prepared code and controller scope are complete. The named batch_dubbing and
modal_control/tests suites passed325tests, with1skip and550subtests, under
oneCPU/512MiB. Fifty tests cover the new transition/ASR components. See
HANDOFF.md and transition_release_scope.json for exact preparation details.
No release or successful model/API run is implied by these checks.
