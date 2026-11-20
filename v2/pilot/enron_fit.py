"""300 real emails for fitting the CONTENT threshold, kept apart from rows.jsonl and from the held-out Enron rows."""
import json, sys
from pathlib import Path
sys.path[:0] = ['v2', '.']
import labeler, build_rows as b, data, sources

data.load_env()
held = [r for r in labeler.read_rows('v2/data/rows.jsonl') if r['source'] == 'enron']
shingled = [sources.shingles(r['text']) for r in held]
fresh = b.sample('LLM-PBE/enron-email', 'train', 490_408, 400, 'enron_fit', lambda r: r['text'])
keep = [r | {'split': 'content_fit'} for r in fresh if not any(sources.near_dup(sources.shingles(r['text']), x) for x in shingled)][:300]
assert len(keep) == 300, len(keep)
Path('v2/data/enron_fit_rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in keep))
print(len(keep), 'rows; dropped as near-duplicates of held-out', len(fresh) - len(keep))
