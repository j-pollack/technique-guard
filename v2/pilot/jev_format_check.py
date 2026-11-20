"""Score the Jev question on the pilot rows and their labels under the final definition."""
import sys
from pathlib import Path
sys.path[:0] = ['v2', '.']
import jev_question as jq
import numpy as np, data, evaluate as ev, labeler
data.load_env()
Q = jq.QUESTION
rows = labeler.read_jsonl(Path('v2/data/pilot2.jsonl'))
L = {r['key']: r for r in labeler.read_jsonl(Path('v2/data/labels_pilot.jsonl')) if r.get('label')}
fmts = {"structured question, object state": jq.state_text}
def grp(r):
    return 'injecagent' if r.get('source', '').startswith('injecagent') else r['set'] + ('/' + r['kind'] if r['set'] != 'pilot' else '')
for name, fn in fmts.items():
    items = [r | {"text": fn(r), "label": int(L[labeler.key(r)]['label'] != 'not_attack')} for r in rows]
    rs, y, s = ev.scored(items, "v2_untuned", Q)
    print(f"\n{name}: scored {len(rs)}, attacks {int(y.sum())}, AUC {ev.auc(s, y):.3f}")
    g = np.array([grp(r) for r in rs])
    for k in sorted(set(g)):
        m = g == k
        a, n = s[m & (y == 1)], s[m & (y == 0)]
        print(f"  {k:28} n {m.sum():3}  attacks {len(a):3}  median attack {np.median(a) if len(a) else float('nan'):.2f}  median non {np.median(n) if len(n) else float('nan'):.2f}")
    for t in (0.3, 0.5, 0.7):
        b = s > t
        print(f"  thr {t}: attacks {int(b[y == 1].sum())}/{int((y == 1).sum())}  non-attacks flagged {int(b[y == 0].sum())}/{int((y == 0).sum())}")
