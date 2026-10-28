"""Build a local review page from measured artifacts; no inference or media jobs."""
import json
from pathlib import Path

HERE = Path(__file__).parent
data = {name: json.loads((HERE / filename).read_text()) for name, filename in
        (("review", "review_data.json"), ("audit", "metrics.json"), ("audio", "audio_checks.json"))}
template = (HERE / "review_template.html").read_text()
(HERE / "review.html").write_text(template.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")))
print(HERE / "review.html")
