"""Offline checks for the Tower-Plus-2B translator wiring. No model, no network.

Run: uv run python test_tower_flow.py
"""
from __future__ import annotations

import pysrt

import translate_utils as tu


class FakeClient:
    """Records prompts and replays one canned response."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[dict] = []

    def generate(self, model, prompt, options=None):
        self.calls.append({"model": model, "prompt": prompt, "options": options or {}})
        return {"response": self.reply}


def make_subs(texts, start_ms=0, step_ms=2000):
    subs = []
    for i, text in enumerate(texts, 1):
        subs.append(pysrt.SubRipItem(
            index=i, start=pysrt.SubRipTime(milliseconds=start_ms + (i - 1) * step_ms),
            end=pysrt.SubRipTime(milliseconds=start_ms + i * step_ms - 200), text=text))
    return subs


def test_template_is_added_for_tower():
    out = tu.apply_chat_template("hello", "tower-plus-2b:latest")
    assert out.startswith("<bos><start_of_turn>user\n"), out
    assert out.endswith("<start_of_turn>model\n"), out
    assert "hello" in out


def test_template_is_added_for_gemma_too():
    assert tu.apply_chat_template("x", "translategemma:4b").startswith("<bos><start_of_turn>user")


def test_unknown_model_is_passed_through():
    assert tu.apply_chat_template("x", "some-chat-model") == "x"


def test_generate_sets_stop_tokens():
    client = FakeClient("Hallo")
    tu.generate(client, "tower-plus-2b:latest", "p", {"temperature": 0.1})
    assert client.calls[0]["options"]["stop"] == ["<end_of_turn>", "<eos>"]


def test_clean_output_removes_leaked_tokens():
    assert tu.clean_output("Hallo Welt<end_of_turn>") == "Hallo Welt"
    assert tu.clean_output("<bos>Hallo<|endoftext|>") == "Hallo"


def test_split_respects_fragment_count():
    pieces = tu.split_sentence_translation("a b c d e f g h", [10, 10, 10])
    assert len(pieces) == 3
    assert " ".join(pieces).split() == "a b c d e f g h".split()


def test_split_returns_none_when_too_short():
    assert tu.split_sentence_translation("a", [5, 5, 5]) is None


def test_sentence_group_sends_one_call_and_splits():
    subs = make_subs(["I think one of the reasons we do science is",
                      "because every now and then, it's pretty rare, but you",
                      "know, every now and then you make a discovery and"])
    client = FakeClient("Ich denke, einer der Gründe, warum wir Wissenschaft betreiben, "
                        "ist, dass es hin und wieder ziemlich selten ist, aber du weißt, "
                        "hin und wieder macht man eine Entdeckung und")
    result = tu._translate_single_group(subs, "en", "de", client)

    assert len(client.calls) == 1, "sentence must be translated in ONE call, not per cue"
    assert "I think one of the reasons we do science is because" in client.calls[0]["prompt"]
    assert set(result) == {1, 2, 3}
    assert all(result[i].strip() for i in result)


def test_speaker_tags_are_preserved():
    subs = make_subs(["[Speaker 2] Hello there, this is one",
                      "[Speaker 2] complete sentence now."])
    client = FakeClient("Hallo zusammen, das ist jetzt ein vollständiger Satz.")
    result = tu._translate_single_group(subs, "en", "de", client)
    assert result[1].startswith("[Speaker 2]")
    assert result[2].startswith("[Speaker 2]")


def test_falls_back_to_line_mode_when_translation_too_short():
    subs = make_subs(["a fairly long first fragment here", "and a second one"])
    # Model answers with a single word: cannot be split, so line mode kicks in.
    client = FakeClient("Ok")
    result = tu._translate_single_group(subs, "en", "de", client)
    assert result == {}, result
    assert len(client.calls) >= 2, "should retry with the line-by-line prompt"


def test_default_model_is_tower():
    assert tu.DEFAULT_MODEL.startswith("tower-plus-2b")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
