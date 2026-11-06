"""Bounded automated content gate; adapters own the reviewed model execution.

Forced alignment alone is not a content check. Require independent ASR,
source-to-translation semantic review, and only then word alignment.
"""
import hashlib
import json
from pathlib import Path
from .passage_batch import digest_file,identity
from .align_continuous_subtitles import tokens


def edit_distance(a,b):
    previous=list(range(len(b)+1))
    for i,left in enumerate(a,1):
        current=[i]
        for j,right in enumerate(b,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(left!=right)))
        previous=current
    return previous[-1]


class ContentVerifier:
    def __init__(self,*,transcribe,review_meaning,align,provider_identity,max_calls,cache_dir):
        if type(max_calls) is not int or max_calls<=0 or not provider_identity:
            raise ValueError('Explicit check call cap and model identities required')
        self.transcribe=transcribe;self.review_meaning=review_meaning;self.align=align
        self.provider_identity=provider_identity;self.max_calls=max_calls
        self.calls=dict(asr=0,meaning=0,alignment=0);self.cache_hits=0
        self.cache=Path(cache_dir);self.cache.mkdir(parents=True,exist_ok=True)

    def call(self,kind,callback,*args):
        if sum(self.calls.values())>=self.max_calls:raise ValueError('Content verification budget exhausted')
        self.calls[kind]+=1
        return callback(*args)

    def __call__(self,unit,take):
        if digest_file(take['file'])!=take['sha256']:raise ValueError('Raw audio changed before verification')
        binding=dict(unit_id=unit['id'],source_id=unit['source_id'],audio_sha256=take['sha256'],
                     text_sha256=hashlib.sha256(take['text'].encode()).hexdigest(),
                     source_text_sha256=hashlib.sha256(unit['source_text'].encode()).hexdigest())
        key=identity(dict(**binding,provider=self.provider_identity));path=self.cache/(key+'.json')
        if path.exists():
            cached=json.loads(path.read_text());receipt=cached['receipt']
            if cached.get('sha256')!=identity(receipt) or any(receipt.get(k)!=v for k,v in binding.items()):
                raise ValueError('Corrupt or mismatched content receipt')
            self.cache_hits+=1;return receipt
        report=dict(**binding,status='failed',speech_coverage_verified=False,meaning_verified=False,
                    provider=self.provider_identity,failures=[])
        def finish():
            temporary=path.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(receipt=report,sha256=identity(report)),ensure_ascii=False,indent=2)+'\n')
            temporary.replace(path);return report
        transcript=self.call('asr',self.transcribe,take['file'],unit.get('target_language'))
        if not isinstance(transcript,str):raise ValueError('ASR adapter must return independently recognized text')
        expected,actual=tokens(take['text']),tokens(transcript)
        report['recognized_text']=transcript
        report['word_error_rate']=edit_distance(expected,actual)/max(1,len(expected))
        # Conservative coverage: imperfect recognitions are reported, not hidden
        # by feeding the expected text to a forced aligner and calling it proof.
        if actual!=expected:
            report['failures'].append('spoken_text_mismatch');return finish()
        report['speech_coverage_verified']=True
        request=dict(source_text=unit['source_text'],target_text=take['text'],
                     source_language=unit.get('source_language'),target_language=unit.get('target_language'),
                     instruction='Check all source meaning, names, quantities, negation and intent. Report omissions, additions and changed meaning; never judge timing or fluency as proof of fidelity.')
        meaning=self.call('meaning',self.review_meaning,request)
        report['meaning_review']=meaning
        if not (isinstance(meaning,dict) and meaning.get('faithful') is True
                and meaning.get('omissions')==[] and meaning.get('additions')==[]
                and meaning.get('contradictions')==[] and isinstance(meaning.get('reason'),str) and meaning['reason'].strip()):
            report['failures'].append('meaning_not_verified');return finish()
        report['meaning_verified']=True
        words=self.call('alignment',self.align,take['file'],take['text'],unit.get('target_language'))
        if not isinstance(words,list) or [t for w in words for t in tokens(w.get('text',''))]!=expected:
            report['failures'].append('alignment_text_mismatch');return finish()
        previous=0.
        for word in words:
            if not previous<=word['start']<word['end']<=take['edges']['raw_seconds']:
                report['failures'].append('invalid_word_times');return finish()
            previous=word['end']
        report.update(status='passed',aligned_words=words)
        return finish()
