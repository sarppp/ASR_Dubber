# Dubbing timing: professional practice and research

Researched 2026-10-29. This is a research checkpoint, not a listening result or
an inference release. No models, media processing, or cloud jobs were run for it.

## Requirement and correction

The user's first priority is unchanged 1.0x playback, matching original talking
intervals and intervening pauses in English/German; expression comes later.
The solution must generalize without manually repairing each video.

Three different quantities must remain separate:

1. Playback multiplier: no accelerating or slowing the completed waveform.
2. Generated articulation: a model can still rush words at playback 1.0x.
3. Speech/pause timeline: equal file lengths do not imply matching speech activity.

Professional dubbing is useful evidence, but is not proof that every original
pause is preserved exactly. We must not promise universal exact timing and
naturalness from current papers, nor silently weaken the user's requirement.

## What professional dubbing does

Netflix's official guidance describes dialogue adaptation, idiomatic wording,
directed performance, and editing against the picture. Start/end mouth movements,
breaths, and reactions matter. Meaning and intent can take priority when perfect
lip synchronization conflicts with them. These are documented studio guidelines,
not a claim that every cinema production follows one identical workflow.
Sources: [English creative guidelines](https://partnerhelp.netflixstudios.com/hc/en-us/articles/11220839130771-Dubbing-Creative-Guidelines-Films-Series-English),
[editing and mixing guidelines](https://partnerhelp.netflixstudios.com/hc/en-us/articles/10258693072403-Dubbing-Creative-Guidelines-Films-Series).

The TACL 2023 study *Dubbing in Practice* examined approximately 319.6 hours of
professional English-to-German/Spanish localization across 54 titles. Its section
4.3 finds dubbers more willing to relax timing than vary speaking rate when
constraints conflict. Character-count matching is also an imperfect duration
proxy. This corrects the earlier implication that cinematic quality necessarily
means identical speech/silence boundaries.
[Paper](https://aclanthology.org/2023.tacl-1.25.pdf).

## Most relevant papers

### Isochrony-Aware Neural Machine Translation for Automatic Dubbing — 2022

Translates with pause markers and controls phrase length. English-to-German and
French experiments make it directly relevant to our language pair. Experimental
pauses are gaps of at least 300 ms; that is not preservation of every micro-pause.
The synthesis/relaxation procedure means this is not proof of exact timing at
unchanged speaking rate. Aggressive length control can drop content and harm
translation quality. Useful component: represent pauses during translation,
rather than discover the mismatch after synthesis.
[Paper](https://www.isca-archive.org/interspeech_2022/tam22_interspeech.pdf).

### Jointly Optimizing Translations and Speech Timing to Improve Isochrony in Automatic Dubbing — 2023

Jointly predicts translated phonemes and durations, then supplies those durations
to FastSpeech2 before audio generation. This is a particularly relevant blueprint
for producing speech to a timing budget instead of stretching a finished clip.
Tests are German-to-English; they do not establish ready English-to-German
support. Duration discretization and quality/timing tradeoffs remain. Forced
phoneme durations can still affect perceived pace. The public test data and
synthesis code are useful references, not evidence of a complete deployable
German solution.
[Paper](https://arxiv.org/pdf/2302.12979),
[test data](https://github.com/amazon-science/iwslt-autodub-task),
[synthesis code](https://github.com/mtresearcher/FastSpeech2).

### End-to-End Multilingual Automatic Dubbing via Duration-based Translation with Large Language Models — 2025

Estimates a target phoneme budget, generates shorter/longer translation
candidates, ranks fidelity, and aligns pauses. Experiments cover English,
Spanish, and Korean, not German. Uses GPT-4o, ElevenLabs, and speech conversion;
complete reproducible model artifacts were not located. Its speech-overlap
metric measures total-duration difference, not interval-by-interval agreement.
No final waveform stretch is specified, but undocumented API settings prevent a
strict native-cadence claim. Useful idea: revise wording before expensive
synthesis. Our adaptation must forbid invented facts or filler added just to
consume time.
[Paper](https://aclanthology.org/2025.emnlp-demos.37.pdf).

### X-Dubbing: Speech-to-Speech Translation under Temporal and Visual Constraints — October 2026 preprint

Combines time-aware translation with fixed-length speech generation conditioned
on source speech/silence masks and video. Feedback from generated speech trains
the translator. This directly addresses local timing, not only clip length.
Reported pause IoU is 0.551; 97.1% satisfy an active-speech-duration tolerance of
plus/minus 20%, which is not exact equality. Evaluation covers French-to-English
single-speaker monologues, not German or arbitrary films. A complete official
code/weight release was not located. Promising architecture, not a ready fix.
[Full methods and results](https://arxiv.org/html/2610.32042v1).

## Additional checks and implementation candidates

| Work | Useful evidence | Limitation for this task |
| --- | --- | --- |
| [Pause-Aware Automatic Dubbing, 2024](https://aclanthology.org/2024.iwslt-1.2.pdf) | Selects translations and language-appropriate pause boundaries. | Computes a speed factor and regenerates TTS; not unchanged default-rate synthesis. German-to-English tests. |
| [PS-TTS, 2026](https://arxiv.org/pdf/2604.09111) | Combines paraphrasing, pauses, and phonetic compatibility. | Explicit internal duration/rate adjustment; up to 60 paraphrase iterations. No verified German or T4 feasibility. |
| [TransVIP, 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/a32539cb16274581a17e679f6046f4bf-Paper-Conference.pdf) | Uses a source activity sequence at 160 ms resolution. | English/French; sometimes introduces pauses to satisfy duration. [Repository](https://github.com/nethermanpro/transvip) requires separately supplied trained checkpoints. |
| [Dub-S2ST, 2025](https://aclanthology.org/2025.findings-emnlp.524.pdf) | Duration-controlled translated speech units; [released code and weights](https://github.com/kaistmm/Dub-S2ST). | Released translation is French-to-English. Default source trimming removes edge silence; not complete video-timeline preservation. |

F5-TTS is a practical duration-conditioned synthesis candidate to investigate,
not a chosen production backend. Its
[`fix_duration` implementation](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/utils_infer.py)
sets a mel-frame budget before vocoding, including reference duration. It does
not enforce internal pause positions. The
[sampler](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/cfm.py)
can raise the requested length to a minimum; actual outputs must be measured.
The upstream wrapper can parallelize chunks, so any adapter must explicitly
serialize. The [German community checkpoint](https://huggingface.co/hvoss-techfak/F5-TTS-German)
is CC-BY-NC-4.0. No inference was performed and no T4 or local 6 GB fit was
verified. A community checkpoint is not equivalent to a validated bilingual
dubbing system.

## Proposed general architecture — engineering inference from the evidence

This is our proposed design, not a result established by one paper:

1. Measure source speech regions and pauses acoustically, with speaker identity
   and confidence. Preserve original coordinates. ASR/subtitle timestamps alone
   are insufficient; don't mistake unvoiced consonants for sentence pauses.
2. Translate with full conversational context and explicit speech/pause budgets.
   Produce a small bounded set of faithful shorter/longer alternatives. Pauses
   should fall at grammatical locations in the target language.
3. Predict candidate durations for the selected language and voice, then reject
   implausible candidates cheaply. Word/character counts alone are inadequate.
4. Generate with duration and, where supported, pause conditioning. Retain raw
   speech at playback 1.0x. Keep linguistic context across rendering blocks.
5. Measure actual speech activity, onset/offset errors, each pause error, content
   preservation, and intelligibility. Inspect distributions and failed examples,
   not only averages or equal file duration. Listen for rushed articulation.
6. If a faithful natural candidate does not fit, report that failure. Do not
   conceal it with stretching, clipping, swallowed pauses, or trailing silence.

The user's strict target remains the target. Speech activity measurement has
finite resolution; report that uncertainty instead of claiming sample-perfect
classification. Meaning may need different word order across languages, so
literal word-for-word pause placement cannot be assumed to be natural.

## Repository consequence and next decision

The existing source provenance, bounded workers, raw-audio retention, and
validation infrastructure remain useful. The scheduler's older 1.15x allowance,
boundary shifts, and early endings do not implement this requirement. The old
source-naturalness evaluation catalog remains disabled.

The strict timeline evaluator, backend capability contract, wording-feedback
builder and serial comparison harness are now implemented with synthetic fixtures;
see the current [handoff checkpoint](HANDOFF.md). Next select a reproducible English/German synthesis
candidate and prepare a small bounded comparison testing wording adaptation and
duration conditioning separately. CPU suffices for timeline logic; synthesis
hardware depends on the selected model. Do not select CPU/GPU from paper quality
alone. Any new inference needs its own concrete controller plan under AGENTS.md;
this research does not authorize or launch it.
