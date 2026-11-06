import json
from pathlib import Path
from batch_dubbing.aligned_finish import write_srt,read_srt
from batch_dubbing.measured_repair import repair,repair_prompt


def fixture(tmp_path):
    source='When we first published our CRISPR research, it was clear that it was transformative and we hoped for an impact.'
    text='Als wir unsere CRISPR-Forschung erstmals veröffentlichten, war klar, dass es sich um eine transformative Technologie handeln würde, und wir hofften natürlich, dass sie auch Auswirkungen haben würde.'
    script=tmp_path/'spoken.srt';write_srt(script,[dict(start=0,end=2,text='Ein anderer Satz.'),dict(start=3,end=13,text=text)])
    output=tmp_path/'output';output.mkdir();report=output/'alignment_report.json'
    report.write_text(json.dumps(dict(failed_placement=dict(index=2,deadline=13.25,proposed_start=3.,native_seconds=12.),source_plan=dict(source_language='en',target_language='de',passages=[dict(index=2,text=text,comparisons={'en':source})]))))
    return script,report,source


def test_only_measured_failure_changes_after_back_translation(tmp_path):
    script,report,source=fixture(tmp_path)
    candidate='Unsere erste CRISPR-Veröffentlichung war bahnbrechend; wir hofften auf Wirkung.'
    result=repair(script,report,translate=lambda prompt:candidate,back_translate=lambda text:source)
    assert result['status']=='estimate_pending_tts'
    assert read_srt(script)[0]['text']=='Ein anderer Satz.'
    assert read_srt(script)[1]['text']==candidate
    assert script.with_suffix('.before-measured-repair.srt').exists()


def test_missing_clause_back_translation_rejects_without_edit(tmp_path):
    script,report,source=fixture(tmp_path);before=script.read_bytes()
    result=repair(script,report,translate=lambda prompt:'CRISPR veränderte die Forschung.',back_translate=lambda text:'CRISPR changed research.')
    assert result['status']=='rejected' and script.read_bytes()==before


def test_source_only_prompt_preserves_qualifiers():
    prompt=repair_prompt('Original source text','en','de')
    assert prompt.endswith('Original source text') and 'time qualifier' in prompt


def test_hymt_uses_its_documented_native_markers():
    from local_mt import apply_chat_template
    prompt=apply_chat_template('SOURCE','hymt2-1.8b:latest')
    assert prompt=='<｜hy_begin▁of▁sentence｜><｜hy_User｜>SOURCE<｜hy_Assistant｜>'
