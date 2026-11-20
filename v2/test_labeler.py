"""Check run_batch's result handling offline: a label, a refusal that goes to Grok, and a failed row left for a rerun."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).parent))
import labeler

rows = [{"text": t, "channel": "USER MESSAGE"} for t in ("labeled", "refused", "failed")]
k = [labeler.key(r) for r in rows]
ok = NS(stop_reason="end_turn", content=[NS(type="text", text=json.dumps({"reasoning": "r", "label": "not_attack"}))])
results = [NS(custom_id=k[0], result=NS(type="succeeded", message=ok)),
           NS(custom_id=k[1], result=NS(type="succeeded", message=NS(stop_reason="refusal", content=[]))),
           NS(custom_id=k[2], result=NS(type="errored"))]


class Batches:
    def create(self, requests):
        assert len(requests) == 3
        return NS(id="b1")

    def retrieve(self, _):
        return NS(processing_status="ended")

    def results(self, _):
        return results


labeler.anthropic.Anthropic = lambda **_: NS(messages=NS(batches=Batches()))
labeler.grok = lambda r: {"reasoning": "g", "label": "guardrail_evasion"}
out = Path(tempfile.mkdtemp()) / "out.jsonl"
labeler.run_batch(rows, out)
got = {r["key"]: r for r in labeler.read_jsonl(out)}
assert got[k[0]]["by"] == labeler.SONNET and got[k[0]]["label"] == "not_attack"
assert got[k[1]]["by"] == labeler.GROK and got[k[1]]["sonnet_refused"] and got[k[1]]["label"] == "guardrail_evasion"
assert k[2] not in got and not Path(f"{out}.batch").exists()
assert [labeler.key(r) for r in labeler.pending(rows, out)] == [k[2]]
sep = out.with_name("sep.jsonl")
sep.write_text(json.dumps({"text": "a\u2028b\x85c"}, ensure_ascii=False) + "\n")
assert labeler.read_rows(str(sep)) == [{"text": "a\u2028b\x85c"}]
print("ok")
