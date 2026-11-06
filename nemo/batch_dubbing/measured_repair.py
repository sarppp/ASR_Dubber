"""One measured, source-based translation correction; never cut or stretch audio."""
import json
import math
from pathlib import Path
from local_mt import LANG_NAMES,OllamaTranslator,_looks_translated
from dub_meaning import compare_back_translation,missing_facts
from batch_dubbing.aligned_finish import read_srt,write_srt,sha


def repair_prompt(source,source_language,target_language):
    return (f'Translate this {LANG_NAMES.get(source_language,source_language)} passage into concise spoken '
            f'{LANG_NAMES.get(target_language,target_language)}. Preserve every source clause, time qualifier, '
            'intention and technical term. Prefer short direct clauses instead of verbose constructions. '
            'Keep technical terminology close to the source. Do not summarize or invent an ending. '
            f'Output only the {LANG_NAMES.get(target_language,target_language)} translation, in one line.\n\n'+source)


def repair(spoken_script,alignment_report,*,translate=None,back_translate=None,guard=None):
    """Return a recorded candidate or a rejection; apply only after all checks pass.

    Default providers are the already installed pipeline adapter and Tower.
    One candidate and one back-translation maximum; heuristics are not semantic
    certification. The caller must synthesize and measure again before publish.
    """
    report=json.loads(Path(alignment_report).read_text());failed=report.get('failed_placement')
    if not failed:return dict(status='not_repairable',reason='No measured native overrun')
    plan=report['source_plan'];part=next(p for p in plan['passages'] if p['index']==failed['index'])
    source=part['comparisons'][plan['source_language']];original=part['text']
    available=failed['deadline']-failed['proposed_start']
    budget=math.floor(len(original)*available/failed['native_seconds']*.98)
    rows=read_srt(spoken_script);row=next(r for r in rows if r['index']==part['index'])
    if row['text']!=original:raise ValueError('Stale timing report does not match current speech script')
    evidence=dict(status='checking',index=part['index'],source=source,original=original,
                  script_before_sha256=sha(spoken_script),alignment_report_sha256=sha(alignment_report),
                  measured_native_seconds=failed['native_seconds'],available_seconds=available,
                  maximum_candidate_characters=budget,quality_limit='Back-translation and fact checks are conservative heuristics, not semantic certification.')
    destination=Path(alignment_report).parent.parent/'measured_translation_repair.json'
    def save():destination.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    prompt=repair_prompt(source,plan['source_language'],plan['target_language']);evidence['prompt']=prompt
    if translate is None:
        adapter=OllamaTranslator(model='hymt2-1.8b:latest')
        try:candidate=adapter.generate(prompt)
        finally:adapter.unload()
    else:candidate=translate(prompt)
    evidence['candidate']=candidate
    problem=(guard(original,candidate,plan['target_language']) if guard else None)
    if not candidate or len(candidate)>budget or candidate==original:
        problem='Candidate does not meet the measured length budget'
    if not _looks_translated(candidate,source,plan['source_language'],plan['target_language']):
        problem='Candidate is not in the target language'
    if missing_facts(source,candidate):problem='Candidate dropped a source number or acronym'
    if problem:evidence.update(status='rejected',reason=problem);save();return evidence
    if back_translate is None:
        translator=OllamaTranslator()
        try:back=translator.translate_sentence(candidate,plan['target_language'],plan['source_language'])
        finally:translator.unload()
    else:back=back_translate(candidate)
    score,problem=compare_back_translation(source,back)
    missing=missing_facts(source,back)
    if missing:problem='Back-translation dropped source facts'
    evidence.update(back_translation=back,similarity=score,missing_facts=missing)
    if problem:evidence.update(status='rejected',reason=problem);save();return evidence
    backup=Path(spoken_script).with_suffix('.before-measured-repair.srt')
    if backup.exists() and sha(backup)!=sha(spoken_script):
        raise ValueError('One correction already applied; refusing an unbounded retry')
    if not backup.exists():backup.write_bytes(Path(spoken_script).read_bytes())
    row['text']=candidate;write_srt(spoken_script,rows)
    evidence.update(status='estimate_pending_tts',script_after_sha256=sha(spoken_script))
    save();return evidence
