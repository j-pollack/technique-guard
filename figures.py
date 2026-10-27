"""Paper figures as static SVG (stdlib only). Data is copied from PAPER.md Appendix C; edit both together.

    python3 figures.py   # writes figures/*.svg
"""

from pathlib import Path

OUT = Path(__file__).with_name("figures")
# Reference dataviz palette: series slot 1 and text tokens, light and dark (an <img> SVG follows the OS theme).
STYLE = """<style>
  svg { --bar: #2a78d6; --ink: #0b0b0b; --muted: #52514e; --grid: #e4e3df; }
  @media (prefers-color-scheme: dark) { svg { --bar: #3987e5; --ink: #f0efec; --muted: #c3c2b7; --grid: #3a3936; } }
  text { font: 13px system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--ink); }
  .muted { fill: var(--muted); font-size: 12px; }
  .title { font-weight: 600; font-size: 14px; }
  .bar { fill: var(--bar); }
  .ci { stroke: var(--ink); stroke-width: 1.5; }
  .grid { stroke: var(--grid); stroke-width: 1; }
</style>"""

# (label, value %, CI low, CI high); CI may be None
BLOCKED = [("All attacks", 92.9, 91.6, 94.1), ("Injection (direct)", 91.4, 88.8, 93.5), ("Indirect injection", 93.3, 89.6, 95.7),
           ("Jailbreak", 92.5, 88.7, 95.1), ("Obfuscation (encoded)", 100, 97.5, 100), ("Adversarial strings", 97.5, 94.9, 98.8),
           ("Multi-turn (n=25)", 64.0, 44.5, 79.8), ("Attacks in benign sets", 79.2, 66.5, 88.0)]
FPR = [("All non-attacks", 5.5, 4.6, 6.4), ("Benign", 2.5, 1.6, 3.7), ("Plain harmful requests", 0.0, 0.0, 0.6),
       ("Persuasion / role-play / translation", 0.3, 0.0, 1.5), ("PAIR prompts", 0.0, 0.0, 2.8),
       ("Jailbreak-styled, not attacks", 23.4, 17.8, 30.2), ("Injection-styled, not attacks", 38.3, 31.1, 45.9),
       ("Multi-turn, not attacks", 10.8, 6.4, 17.7)]
TARGETS = ["Grok 4.7", "Qwen 3.8 Max", "Qwen 3.8 Flash", "Gemini 4 Argon", "Gemini 3.8 Flash",
           "DeepSeek V4.1 Flash", "DeepSeek V4 Pro", "GLM 5.3 Flash", "GLM 5.3", "All"]
UNGUARDED = [0.9, 4.2, 8.1, 7.2, 7.0, 9.3, 20.6, 15.1, 13.1, 9.5]
GUARDED = [0.5, 1.1, 1.5, 1.5, 1.7, 1.5, 1.8, 1.0, 1.7, 1.4]


def panel(rows, x0, y0, label_w, plot_w, vmax, title, ticks, show_labels=True):
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
        if show_labels:
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
    ticks = [0, 5, 10, 15, 20, 25]
    a, h = panel([(t, v, None, None) for t, v in zip(TARGETS, UNGUARDED)], 0, 0, 140, 210, 25, "Without the guard", ticks)
    b, _ = panel([(t, v, None, None) for t, v in zip(TARGETS, GUARDED)], 400, 0, 0, 210, 25, "With the guard", ticks, show_labels=False)
    svg("attack_success.svg", a + b, 680, h, "Share of replayed attacks that succeed per target model, without and with the guard")
    assert len(TARGETS) == len(UNGUARDED) == len(GUARDED)
    print("wrote", *sorted(f.name for f in OUT.glob("*.svg")))


if __name__ == "__main__":
    main()
