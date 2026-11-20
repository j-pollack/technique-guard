import sys, json, random, collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, '/Users/jonathan/Code/prompt-guard')
import anthropic, data, labeler
S = Path(__file__).parent
SYSTEM = (S / 'nist_prompt.txt').read_text().strip()
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["reasoning", "channel", "goal", "label"], "properties": {
    "reasoning": {"type": "string", "description": "One or two sentences."},
    "channel": {"type": "string", "enum": ["direct", "indirect"]},
    "goal": {"type": "string", "enum": ["misuse", "privacy", "integrity", "availability", "none"]},
    "label": {"type": "string", "enum": ["jailbreak", "direct_prompt_injection", "prompt_extraction", "indirect_prompt_injection", "not_attack"]}}}
data.load_env()
rows = data.rows('spent_test')
strata = collections.defaultdict(list)
for r in rows: strata[(r['source'], r['family'], r['label'])].append(r)
rng = random.Random(0)
pick = [x for k in sorted(strata, key=str) for x in rng.sample(strata[k], min(3, len(strata[k])))]
print(len(rows), 'rows;', len(strata), 'strata;', len(pick), 'picked', flush=True)
client = anthropic.Anthropic(max_retries=8)
def ask(r):
    try:
        m = client.messages.create(model=labeler.MODEL, max_tokens=2000, system=SYSTEM,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": json.dumps({"TEXT": r['text']})}])
        if m.stop_reason == 'refusal': return {"error": "refusal"}, m.usage
        return json.loads(next(b.text for b in m.content if b.type == 'text')), m.usage
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}, None
with ThreadPoolExecutor(8) as p: out = list(p.map(ask, pick))
with open(S / 'pilot.jsonl', 'w') as f:
    for r, (o, u) in zip(pick, out):
        f.write(json.dumps({k: r[k] for k in ('text', 'source', 'family', 'label', 'label_from', 'dataset_label')} | {"new": o}) + '\n')
tok = [u for _, u in out if u]
print('input', sum(u.input_tokens for u in tok), 'output', sum(u.output_tokens for u in tok))
print('errors', collections.Counter(o.get('error', 'ok')[:60] for o, _ in out))
