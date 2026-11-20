import sys, json, collections
from pathlib import Path
sys.path.insert(0, '/Users/jonathan/Code/prompt-guard')
import numpy as np, data, jev, evaluate as ev
from questions import STAGES, ablated
data.load_env()
p = Path('/Users/jonathan/Code/prompt-guard/data/benign_resource.json')
rows = json.loads(p.read_text())
moves = {"Preheat the oven": "requests", "Get 20% off select": "requests", "Save $15 on your first": "requests", "Would love it if Dell": "content"}
moved = 0
for r in rows:
    for k, v in moves.items():
        if r['filler'].startswith(k) and r['kind'] != v: r['kind'] = v; moved += 1
assert moved == 4, moved
p.write_text(json.dumps(rows, ensure_ascii=False, indent=0))
print(collections.Counter(r['kind'] for r in rows))
Qs = {"final": STAGES["final"], "bare": STAGES["bare"]}
nd = ablated("no_data_injection")
dev = data.rows("dev")
cache = {(json.loads(l)["t"], json.loads(l)["q"]) for l in jev.ANSWERS.read_text().splitlines()}
k = jev.qkey(nd); missing = sum((jev.sha(i["text"]), k) not in cache for i in dev)
print('no_data_injection dev answers missing', missing)
thr = {"final": ev.dev_threshold("final", STAGES["final"]), "bare": ev.dev_threshold("bare", STAGES["bare"])}
if missing == 0:
    Qs["no_data_injection"] = nd; thr["no_data_injection"] = ev.dev_threshold("no_data_injection", nd)
print('thresholds', thr)
out = {}
for n, q in Qs.items():
    rs, y, s = ev.scored(rows, n, q)
    assert len(rs) == len(rows), (n, len(rs))
    b = s > thr[n]; out[n] = {"thr": thr[n], "scores": s.tolist()}
    kinds = np.array([r['kind'] for r in rs])
    print(f"\n{n}: blocked {b.sum()}/{len(b)} = {b.mean():.1%} {ev.wilson(int(b.sum()), len(b))}; median score {np.median(s):.2f}")
    for kd in ('content', 'requests'):
        m = kinds == kd; print(f"  {kd}: {b[m].sum()}/{m.sum()} = {b[m].mean():.1%} {tuple(round(x,3) for x in ev.wilson(int(b[m].sum()), int(m.sum())))}; median {np.median(s[m]):.2f}")
    for r, sc, bl in sorted(zip(rs, s, b), key=lambda x: -x[1]):
        if bl and n != 'bare': print(f"   {sc:.2f} [{r['kind'][0]} t{r['template']}] {r['filler'][:105]}")
    print('  by template', {t: int(b[[r['template'] == t for r in rs]].sum()) for t in range(17)})
Path(sys.argv[0]).with_name('benign_scores.json').write_text(json.dumps(out))
