"""A separate threshold for CONTENT rows, fit on dev, the harmless synthetic rows, and 300 new Enron emails, checked on the spent test.
Held-out, new samples, and traffic stay unread. Each channel flags 5% of its own non-attacks in the fit rows."""
import collections, sys
from pathlib import Path
sys.path[:0] = ['v2', '.']
import numpy as np
import jev_four as f, labeler, evaluate as ev, data, jev

data.load_env()
rows = [r for r in labeler.read_rows('v2/data/rows.jsonl') if r['split'] in ('dev', 'spent_test') or (r['split'] == 'synthetic' and r['channel'] == 'CONTENT')]
L = {a['key']: a for a in labeler.read_jsonl(Path('v2/data/labels.jsonl'))}
one = {x['key']: x['noul'] for x in labeler.read_jsonl(Path('v2/data/jev_scores.jsonl'))}
enron = labeler.read_rows('v2/data/enron_fit_rows.jsonl')
L |= {a['key']: a for a in labeler.read_jsonl(Path('v2/data/enron_fit_labels.jsonl'))}
rows += enron
rows = [r for r in rows if L[labeler.key(r)].get('label')]
content = [r for r in rows if r['channel'] == 'CONTENT']
items = [{'text': f.jq.state_text(r)} for r in content]
old = jev.fill([{'text': f.jq.state_text(r)} for r in enron], {'v2_untuned': f.jq.QUESTION})
one |= {labeler.key(r): old[(f.jq.state_text(r), 'v2_untuned')]['noul'] for r in enron}
A = jev.fill(items, {f.INJECTION: f.QUESTIONS[f.INJECTION]})
inj = {labeler.key(r): A[(i['text'], f.INJECTION)]['noul'] for r, i in zip(content, items)}
y = lambda x: np.array([int(L[labeler.key(r)]['label'] != 'not_attack') for r in x])
pick = lambda split, ch: [r for r in rows if r['channel'] == ch and r['split'] in split]
FIT = ('dev', 'synthetic', 'content_fit')
fit_c, fit_u = pick(FIT, 'CONTENT'), pick(('dev',), 'USER MESSAGE')
print('fit CONTENT', len(fit_c), 'non-attacks', int((y(fit_c) == 0).sum()), '| fit USER', len(fit_u))
s1 = lambda x: np.array([one[labeler.key(r)] for r in x])
s2 = lambda x: np.array([inj[labeler.key(r)] for r in x])
tu = float(ev.threshold_at(s1(fit_u), y(fit_u)))
for name, s in (('one Noul', s1), ('injection Noul', s2)):
    tc = float(ev.threshold_at(s(fit_c), y(fit_c)))
    print(f'\n{name}: CONTENT thr {tc:.3f}, USER thr {tu:.3f}')
    for split in ('dev', 'spent_test'):
        c, u = pick((split,), 'CONTENT'), pick((split,), 'USER MESSAGE')
        cy, uy = y(c), y(u); cb = s(c) > tc; ub = s1(u) > tu
        tp, fp = (cb & (cy == 1)).sum(), (cb & (cy == 0)).sum()
        agree = ((cb == (cy == 1)).sum() + (ub == (uy == 1)).sum()) / (len(c) + len(u))
        print(f'  {split}: CONTENT n {len(c)} AUC {ev.auc(s(c), cy):.3f} caught {tp}/{cy.sum()} flags {fp}/{(cy == 0).sum()} | all rows: agree {agree:.1%} caught {tp + (ub & (uy == 1)).sum()}/{cy.sum() + uy.sum()} flags {fp + (ub & (uy == 0)).sum()}/{(cy == 0).sum() + (uy == 0).sum()}')
        t = collections.defaultdict(lambda: [0, 0])
        for r, b in zip(c, cb):
            k = (r['source'], L[labeler.key(r)]['label']); t[k][0] += 1; t[k][1] += int(b)
        print('   ', {f'{a}/{b}': f'{v[1]}/{v[0]}' for (a, b), v in sorted(t.items())})
    for sp in ('synthetic', 'content_fit'):
        x = pick((sp,), 'CONTENT'); print(f'   {sp} flagged', int((s(x) > tc).sum()), 'of', len(x))
