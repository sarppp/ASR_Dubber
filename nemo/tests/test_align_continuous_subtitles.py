import pytest
from batch_dubbing.align_continuous_subtitles import aligned_spans,render,tokens


def test_word_boundaries_replace_old_cue_times_without_changing_text():
    rows=[dict(index=1,timing='old',text='Ich denke,'),dict(index=2,timing='old',text='weiß ich.')]
    words=[dict(text=w,start=s,end=e) for w,s,e in [('Ich',.1,.2),('denke',.2,.7),('weiß',.8,1.),('ich',1.,1.2)]]
    spans=aligned_spans(rows,words,2)
    assert spans[0][0]==pytest.approx(.02)
    assert spans[0][1]==spans[1][0]==pytest.approx(.72)
    assert spans[1][1]==pytest.approx(1.32)
    data=render(rows,spans).decode('utf-8-sig')
    assert 'Ich denke,' in data and 'weiß ich.' in data
    assert '00:00:00,020 --> 00:00:00,720' in data
    english=[dict(index=1,text='I think,'),dict(index=2,text='I know.')]
    assert render(english,spans).decode('utf-8-sig').splitlines()[1]==data.splitlines()[1]


@pytest.mark.parametrize('words',[
    [dict(text='wrong',start=0,end=1)],
    [dict(text='Hallo',start=.3,end=.2)],
    [dict(text='Hallo',start=0,end=10)],
    [dict(text='Hallo',start=float('nan'),end=1)],
])
def test_invalid_alignment_rejected(words):
    with pytest.raises(ValueError):aligned_spans([dict(text='Hallo')],words,2)


def test_german_punctuation_normalization():
    assert tokens('Gründe, weißt CRISPR. Gefühl!')==['gründe','weisst','crispr','gefühl']
