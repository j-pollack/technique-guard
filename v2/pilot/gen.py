import sys, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, '/Users/jonathan/Code/prompt-guard')
import anthropic, data, labeler
S = Path(__file__).parent
HOLE = '<Attacker Instruction>'
SYSTEM = (S / 'gen_prompt.txt').read_text().strip()
L = {"type": "array", "items": {"type": "string"}}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["content", "requests"], "properties": {"content": L, "requests": L}}
cases = [json.loads(l) for l in open(S / 'injecagent_dl/user_cases.jsonl')]
T = sorted({c['Tool Response Template'] for c in cases})
assert len(T) == 17 and all(t.count(HOLE) == 1 for t in T)
data.load_env()
client = anthropic.Anthropic(max_retries=8)
def gen(t):
    m = client.messages.create(model=labeler.MODEL, max_tokens=2000, system=SYSTEM,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": t.replace(HOLE, '<FIELD>')}])
    assert m.stop_reason != 'refusal', t[:80]
    return json.loads(next(b.text for b in m.content if b.type == 'text'))
with ThreadPoolExecutor(6) as p: out = list(p.map(gen, T))
rows = []
for i, (t, o) in enumerate(zip(T, out)):
    assert len(o['content']) == 5 and len(o['requests']) == 5, (i, len(o['content']), len(o['requests']))
    for kind in ('content', 'requests'):
        for f in o[kind]:
            rows.append({"text": '"' + t.replace(HOLE, f) + '"', "filler": f, "kind": kind, "template": i, "label": 0,
                         "source": "benign_resource", "family": "benign_resource", "label_from": "construction"})
Path('/Users/jonathan/Code/prompt-guard/data/benign_resource.json').write_text(json.dumps(rows, ensure_ascii=False, indent=0))
print(len(rows), 'rows')
for i, t in enumerate(T):
    print(f'\n[{i}] {t[:70]}')
    for r in rows:
        if r['template'] == i: print(f"  {r['kind'][0]} {r['filler']}")
