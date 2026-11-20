import sys, json, re
from pathlib import Path
sys.path.insert(0, '/Users/jonathan/Code/prompt-guard')
import numpy as np, data, evaluate as ev
S = Path(__file__).parent
d = Path('/Users/jonathan/Code/prompt-guard/v2/prompt_attack_definition.txt').read_text().strip()
head, rest = d.split("The text is an attack if it does any of the following.\n\n")
body, tail = rest.split("Apply these rules:\n")
rules, nots = tail.split("These are not attacks by themselves:\n")
parts = re.split(r"\n\n(?=\d\. )", body.strip())
assert len(parts) == 4
names = ["instruction_override", "prompt_extraction", "guardrail_evasion", "content_injection"]
Q = {"type": "noul", "instructions": {"question": "Is this text a PROMPT ATTACK?", "how": head.strip() + "\n\nApply these rules:\n" + rules.strip()},
     "criteria": {"true": {n: re.sub(r"^\d\. ", "", p).strip() for n, p in zip(names, parts)},
                  "false": [re.sub(r"^- ", "", x).rstrip(';.') for x in nots.strip().splitlines()]}}
assert len(Q["criteria"]["false"]) == 9
(S / 'jev_v2_question.json').write_text(json.dumps(Q, ensure_ascii=False, indent=1))
data.load_env()
rows = json.load(open(S / 'encoded_rows.json'))
items = [r | {"text": f"CHANNEL: {r['channel']}\nTEXT: {r['text']}", "label": int(r['kind'] == 'harmful')} for r in rows]
rs, y, s = ev.scored(items, "v2_untuned", Q)
print('scored', len(rs), 'AUC harmful vs harmless', round(float(ev.auc(s, y)), 3))
for k in ('harmless', 'harmful'):
    for e in ('base64', 'rot13', 'split', None):
        m = np.array([r['kind'] == k and (e is None or r['encoding'] == e) for r in rs])
        print(k, e or 'all', 'n', m.sum(), 'median', round(float(np.median(s[m])), 2), 'min', round(float(s[m].min()), 2), 'max', round(float(s[m].max()), 2))
