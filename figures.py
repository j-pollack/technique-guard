"""Paper figures as static SVG (stdlib only). Data is copied from PAPER.md Appendix C; edit both together.

    python3 figures.py   # writes figures/*.svg
"""

from pathlib import Path

OUT = Path(__file__).with_name("figures")
# Reference dataviz palette: series slot 1 and text tokens, light and dark (an <img> SVG follows the OS theme).
STYLE = """<style>
  svg { --bar: #2a78d6; --bar2: #eb6834; --ink: #0b0b0b; --muted: #52514e; --grid: #e4e3df; }
  @media (prefers-color-scheme: dark) { svg { --bar: #3987e5; --bar2: #d95926; --ink: #f0efec; --muted: #c3c2b7; --grid: #3a3936; } }
  text { font: 13px system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--ink); }
  .muted { fill: var(--muted); font-size: 12px; }
  .hdr { font-size: 10.5px; }
  .title { font-weight: 600; font-size: 14px; }
  .bar { fill: var(--bar); }
  .bar2 { fill: var(--bar2); }
  .ci { stroke: var(--ink); stroke-width: 1.5; }
  .grid { stroke: var(--grid); stroke-width: 1; }
__HEAT__
</style>"""

# (label, value %, CI low, CI high); CI may be None
BLOCKED = [("All attacks", 92.9, 91.6, 94.1), ("Injection (direct)", 91.4, 88.8, 93.5), ("Indirect injection", 93.3, 89.6, 95.7),
           ("Jailbreak", 92.5, 88.7, 95.1), ("Obfuscation (encoded)", 100, 97.5, 100), ("Adversarial strings", 97.5, 94.9, 98.8),
           ("Multi-turn (n=25)", 64.0, 44.5, 79.8), ("Attacks in benign sets", 79.2, 66.5, 88.0)]
FPR = [("All non-attacks", 5.5, 4.6, 6.4), ("Benign", 2.5, 1.6, 3.7), ("Plain harmful requests", 0.0, 0.0, 0.6),
       ("Persuasion / role-play / translation", 0.3, 0.0, 1.5), ("PAIR prompts", 0.0, 0.0, 2.8),
       ("Jailbreak-styled, not attacks", 23.4, 17.8, 30.2), ("Injection-styled, not attacks", 38.3, 31.1, 45.9),
       ("Multi-turn, not attacks", 10.8, 6.4, 17.7)]
# Sequential blue ramp (steps 100..700); cell text flips to white on the dark half.
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b"]
HEAT = "\n".join(f"  .h{i} {{ fill: {c}; }} .t{i} {{ fill: {'#ffffff' if i >= 6 else '#0b0b0b'}; }}" for i, c in enumerate(RAMP)) + \
    "\n  @media (prefers-color-scheme: dark) {\n" + \
    "\n".join(f"    .h{i} {{ fill: {c}; }} .t{i} {{ fill: {'#ffffff' if i < 6 else '#0b0b0b'}; }}" for i, c in enumerate(reversed(RAMP))) + "\n  }"
STYLE = STYLE.replace("__HEAT__", HEAT)
TARGETS = ["Grok 4.7", "Qwen 3.8 Max", "Qwen 3.8 Flash", "Gemini 4 Argon", "Gemini 3.8 Flash",
           "DeepSeek V4.1 Flash", "DeepSeek V4 Pro", "GLM 5.3 Flash", "GLM 5.3", "All"]
UNGUARDED = [0.9, 4.2, 8.1, 7.2, 7.0, 9.3, 20.6, 15.1, 13.1, 9.5]
GUARDED = [0.5, 1.1, 1.5, 1.5, 1.7, 1.5, 1.8, 1.0, 1.7, 1.4]
# (family, success % without the guard per target, with the guard, graded pairs per target); None = no graded pairs
FAMILY_SUCCESS = [
    ("HackAPrompt", [0.0, 16.0, 26.4, 6.1, 16.0, 25.2, 20.6, 39.7, 39.7], [0.0, 6.8, 7.2, 1.5, 6.8, 6.8, 6.8, 7.6, 12.2], [31, 31, 30, 31, 31, 31, 31, 31, 31]),
    ("Tensor Trust hijack", [0.0, 1.7, 11.5, 0.0, 5.9, 0.8, 1.7, 0.8, 5.9], [0.0, 1.7, 0.9, 0.0, 0.8, 0.8, 1.7, 0.8, 0.8], [24, 24, 23, 24, 24, 24, 24, 24, 24]),
    ("Tensor Trust extract", [0.0, 9.9, 14.6, 16.4, 9.9, 13.1, 13.1, 16.4, 20.4], [0.0, 0.0, 0.0, 3.3, 0.0, 0.0, 0.0, 0.0, 0.7], [32, 32, 29, 32, 32, 32, 32, 32, 32]),
    ("InjecAgent direct harm", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.3, 0.0, 5.2], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0], [25, 25, 25, 25, 25, 25, 25, 25, 25]),
    ("InjecAgent data stealing", [2.7, 0.0, 0.0, 0.0, 0.0, 0.0, 3.4, 2.7, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0], [38, 38, 38, 38, 38, 38, 38, 38, 38]),
    ("WildGuardMix jailbreak", [8.6, 0.0, 7.0, 22.7, 21.1, 8.6, 15.6, 7.0, 7.0], [8.6, 0.0, 7.0, 8.6, 7.0, 8.6, 8.6, 0.0, 0.0], [15, 15, 15, 15, 15, 15, 15, 15, 15]),
    ("WildJailbreak adversarial", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [5, 5, 5, 5, 5, 5, 5, 5, 5]),
    ("MHJ multi-turn (3–4 per cell)", [0.0, 0.0, 47.1, 64.0, 47.1, 32.0, 64.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [4, 4, 3, 4, 3, 4, 4, 4, 3]),
    ("SORRY encodings", [None, None, None, 0.0, 4.2, None, None, 0.0, None], [None, None, None, 0.0, 0.0, None, None, 0.0, None], [0, 0, 0, 5, 24, 0, 0, 1, 0]),
    ("GCG suffixes", [0.0, 3.1, 3.2, 9.4, 6.2, 9.7, 9.4, 9.4, 9.4], [0.0, 0.0, 0.0, 3.1, 3.1, 0.0, 0.0, 0.0, 0.0], [32, 32, 31, 32, 32, 31, 32, 32, 32]),
    ("Random-search suffixes", [0.0, 2.6, 0.0, 2.6, 2.7, 14.3, 83.3, 44.7, 18.4], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [38, 38, 33, 38, 37, 35, 36, 38, 38]),
]
SHORT = [("Grok", "4.7"), ("Qwen", "Max"), ("Qwen", "Flash"), ("Gemini", "Argon"), ("Gemini", "3.8 Flash"),
         ("DeepSeek", "V4.1 Flash"), ("DeepSeek", "V4 Pro"), ("GLM", "5.3 Flash"), ("GLM", "5.3")]


def panel(rows, x0, y0, label_w, plot_w, vmax, title, ticks):
    """Horizontal bars with optional CI whiskers; returns (svg parts, height)."""
    row_h, bar_h, top = 26, 14, 28
    out = [f'<text class="title" x="{x0}" y="{y0 + 14}">{title}</text>']
    px = lambda v: x0 + label_w + v / vmax * plot_w
    h = top + row_h * len(rows)
    for t in ticks:
        out.append(f'<line class="grid" x1="{px(t):.1f}" x2="{px(t):.1f}" y1="{y0 + top - 4}" y2="{y0 + h}"/>')
        out.append(f'<text class="muted" x="{px(t):.1f}" y="{y0 + h + 16}" text-anchor="middle">{t:g}%</text>')
    for i, (label, v, lo, hi) in enumerate(rows):
        y = y0 + top + i * row_h
        out.append(f'<text x="{x0 + label_w - 10}" y="{y + bar_h - 2}" text-anchor="end">{label}</text>')
        w = max(px(v) - px(0), 0)
        # 4px rounded data end, square at the baseline
        out.append(f'<path class="bar" d="M{px(0):.1f},{y} h{max(w - 4, 0):.1f} q4,0 4,4 v{bar_h - 8} q0,4 -4,4 h{-max(w - 4, 0):.1f} z"/>'
                   if w >= 4 else f'<rect class="bar" x="{px(0):.1f}" y="{y}" width="{w:.1f}" height="{bar_h}"/>')
        if lo is not None:
            cy = y + bar_h / 2
            out.append(f'<line class="ci" x1="{px(lo):.1f}" x2="{px(hi):.1f}" y1="{cy}" y2="{cy}"/>')
        end = max(px(v), px(hi) if hi is not None else 0)
        out.append(f'<text class="muted" x="{end + 6:.1f}" y="{y + bar_h - 2}">{v:g}%</text>')
    return out, h + 24


def grouped(x0, y0, label_w, plot_w, vmax, ticks):
    """Two bars per target: without the guard (orange) above with the guard (blue), one shared axis."""
    row_h, bar_h, top = 40, 13, 48
    px = lambda v: x0 + label_w + v / vmax * plot_w
    out = [f'<rect class="bar2" x="{x0 + label_w}" y="{y0 + 6}" width="12" height="12" rx="2"/>',
           f'<text x="{x0 + label_w + 18}" y="{y0 + 16}">Without the guard</text>',
           f'<rect class="bar" x="{x0 + label_w + 160}" y="{y0 + 6}" width="12" height="12" rx="2"/>',
           f'<text x="{x0 + label_w + 178}" y="{y0 + 16}">With the guard</text>']
    h = top + row_h * len(TARGETS)
    for t in ticks:
        out.append(f'<line class="grid" x1="{px(t):.1f}" x2="{px(t):.1f}" y1="{y0 + top - 4}" y2="{y0 + h}"/>')
        out.append(f'<text class="muted" x="{px(t):.1f}" y="{y0 + h + 16}" text-anchor="middle">{t:g}%</text>')
    for i, (t, a, b) in enumerate(zip(TARGETS, UNGUARDED, GUARDED)):
        y = y0 + top + i * row_h
        out.append(f'<text x="{x0 + label_w - 10}" y="{y + bar_h + 5}" text-anchor="end">{t}</text>')
        for j, (v, cls) in enumerate(((a, "bar2"), (b, "bar"))):
            yy = y + j * (bar_h + 2)  # 2px surface gap between the pair
            w = px(v) - px(0)
            out.append(f'<path class="{cls}" d="M{px(0):.1f},{yy} h{max(w - 4, 0):.1f} q4,0 4,4 v{bar_h - 8} q0,4 -4,4 h{-max(w - 4, 0):.1f} z"/>'
                       if w >= 4 else f'<rect class="{cls}" x="{px(0):.1f}" y="{yy}" width="{w:.1f}" height="{bar_h}"/>')
            out.append(f'<text class="muted" x="{px(v) + 6:.1f}" y="{yy + bar_h - 2}">{v:g}%</text>')
    return out, h + 24


def heatmap(x0, y0, label_w, col, idx, title):
    """Family x target grid of success %, binned onto the sequential ramp (0-90%)."""
    row_h, top = 26, 52
    out = [f'<text class="title" x="{x0}" y="{y0 + 14}">{title}</text>']
    for j, (a, b) in enumerate(SHORT):
        cx = x0 + label_w + j * col + col / 2
        out.append(f'<text class="muted hdr" x="{cx}" y="{y0 + 32}" text-anchor="middle">{a}</text>')
        out.append(f'<text class="muted hdr" x="{cx}" y="{y0 + 46}" text-anchor="middle">{b}</text>')
    for i, row in enumerate(FAMILY_SUCCESS):
        y = y0 + top + i * row_h
        out.append(f'<text x="{x0 + label_w - 10}" y="{y + 17}" text-anchor="end">{row[0]}</text>')
        for j, v in enumerate(row[idx]):
            x = x0 + label_w + j * col
            if v is None:
                out.append(f'<text class="muted" x="{x + col / 2}" y="{y + 17}" text-anchor="middle">–</text>')
                continue
            k = min(int(v / 90 * (len(RAMP) - 1) + 0.999), len(RAMP) - 1) if v else None
            if k is not None:  # 2px surface gap between cells
                out.append(f'<rect class="h{k}" x="{x + 1}" y="{y + 1}" width="{col - 2}" height="{row_h - 2}" rx="2"/>')
            out.append(f'<text class="{f"t{k}" if k is not None else "muted"}" x="{x + col / 2}" y="{y + 17}" text-anchor="middle">{v:.0f}</text>')
    return out, top + row_h * len(FAMILY_SUCCESS) + 8


def svg(name, parts, w, h, desc):
    body = "\n".join(parts)
    (OUT / name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">'
                            f"<title>{desc}</title>{STYLE}\n{body}\n</svg>\n")


def main():
    OUT.mkdir(exist_ok=True)
    p, h = panel(BLOCKED, 0, 0, 190, 420, 100, "Attacks blocked, by family (95% CI)", [0, 25, 50, 75, 100])
    svg("attacks_blocked.svg", p, 680, h, "Share of test attacks the guard blocks, by family")
    p, h = panel(FPR, 0, 0, 250, 360, 50, "Non-attacks wrongly blocked, by group (95% CI)", [0, 10, 20, 30, 40, 50])
    svg("false_positives.svg", p, 680, h, "Share of test non-attacks the guard blocks, by group")
    p, h = grouped(0, 0, 150, 460, 25, [0, 5, 10, 15, 20, 25])
    svg("attack_success.svg", p, 680, h, "Share of replayed attacks that succeed per target model, without and with the guard")
    a, h = heatmap(0, 0, 200, 58, 1, "Without the guard: % of attacks that succeed")
    b, _ = heatmap(0, h + 16, 200, 58, 2, "With the guard")
    svg("success_by_family.svg", a + b, 724, 2 * h + 16, "Share of replayed attacks that succeed, by attack family and target model, without and with the guard")
    assert len(TARGETS) == len(UNGUARDED) == len(GUARDED) and all(len(r[1]) == len(r[2]) == len(SHORT) for r in FAMILY_SUCCESS)
    print("wrote", *sorted(f.name for f in OUT.glob("*.svg")))


if __name__ == "__main__":
    main()
