"""Jev against the labels on dev and the spent test only. Held-out, new samples, and traffic stay unread."""
import sys, collections
from pathlib import Path
sys.path[:0] = ['v2', '.']
import numpy as np, labeler, evaluate as ev
rows = labeler.read_rows('v2/data/rows.jsonl')
L = {}
for a in labeler.read_jsonl(Path('v2/data/labels.jsonl')): L[a['key']] = a
S = {x['key']: x['noul'] for x in labeler.read_jsonl(Path('v2/data/jev_scores.jsonl'))}
print('rows', len(rows), 'scored', len(S), 'labels by', collections.Counter(str(a.get('by')) for a in L.values()), 'unlabeled', collections.Counter(str(a.get('error'))[:12] for a in L.values() if not a.get('label')))
def view(split):
    x = [r for r in rows if r['split'] == split and L[labeler.key(r)].get('label') and labeler.key(r) in S]
    y = np.array([int(L[labeler.key(r)]['label'] != 'not_attack') for r in x]); s = np.array([S[labeler.key(r)] for r in x])
    return x, y, s
dx, dy, ds = view('dev')
thr = float(ev.threshold_at(ds, dy)); print('dev threshold', thr)
for split in ('dev', 'spent_test'):
    x, y, s = view(split); b = s > thr
    tp, fp, fn, tn = int((b & (y == 1)).sum()), int((b & (y == 0)).sum()), int((~b & (y == 1)).sum()), int((~b & (y == 0)).sum())
    print(f"\n{split}: n {len(x)} attacks {y.sum()} AUC {ev.auc(s, y):.3f}  agree {(tp+tn)/len(x):.1%}  attacks caught {tp}/{tp+fn} = {tp/(tp+fn):.1%}  false flags {fp}/{fp+tn} = {fp/(fp+tn):.1%}")
    t = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r, yy, bb in zip(x, y, b):
        k = (r['source'], r['family']); v = t[k]; v[0] += int(yy); v[1] += int(yy and bb); v[2] += int(not yy); v[3] += int((not yy) and bb)
    if split == 'spent_test':
        for k, v in sorted(t.items(), key=lambda p: (-(p[1][0] - p[1][1]), -p[1][3])):
            if v[0] >= 8 or v[3] >= 4: print(f"  {k[0]:18}{k[1]:22} attacks {v[1]:4}/{v[0]:<4} non-attack flagged {v[3]:4}/{v[2]}")
    by = collections.Counter(); c = collections.Counter()
    for r, yy, bb in zip(x, y, b):
        a = L[labeler.key(r)]
        if yy: c[a['label']] += 1; by[a['label']] += int(bb)
    print('  by class', {k: f"{by[k]}/{c[k]}" for k in c})
    gr = np.array([L[labeler.key(r)]['by'] != labeler.SONNET for r in x])
    print(f"  on Grok-labeled rows n {gr.sum()}: agree {((b == (y == 1))[gr]).mean():.1%}; Sonnet-labeled n {(~gr).sum()}: {((b == (y == 1))[~gr]).mean():.1%}")
