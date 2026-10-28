"""Generate docs/architecture.drawio (editable in app.diagrams.net) and docs/architecture.svg from one spec."""

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent
W, H = 1180, 565

# lane: (x, y, w, h, title, fill)
LANES = [
    (10, 10, 1160, 240, "OFFLINE  ·  build once per corpus", "#f5f5f5"),
    (10, 265, 1160, 290, "ONLINE  ·  per question, CPU only (165 ms p50)", "#eef4fb"),
]
# group: (x, y, w, h, title) - a titled container that edges can point to
GROUPS = {"score": (690, 290, 240, 205, "Score chunks of routed contract(s)")}
# box id: (x, y, w, h, title, subtitle, fill, stroke)
BOXES = {
    "corpus":  (30, 70, 170, 80, "Contracts", "LegalBench-RAG-mini\n72 docs · 8.7M chars", "#ffffff", "#666666"),
    "chunk":   (250, 45, 190, 70, "Heading-aware chunker", "≤500 chars, no overlap\nexact char offsets", "#fff2cc", "#d6b656"),
    "rindex":  (250, 160, 190, 70, "Router index", "BM25 over contract\nname + first 2k chars", "#e1d5e7", "#9673a6"),
    "ctx":     (480, 45, 190, 70, "Context prefix", "\"title | section\"\n+ chunk text", "#fff2cc", "#d6b656"),
    "enc":     (710, 35, 200, 100, "Encode chunks", "bge-small (dense)\nSPLADE++ (sparse)\nKaggle T4, ~6 min · 26 MB", "#dae8fc", "#6c8ebf"),
    "q":       (30, 355, 170, 90, "Question", "\"Consider the merger\nbetween X and Y; where\nis the No-Shop clause?\"", "#ffffff", "#666666"),
    "route":   (250, 345, 210, 110, "1 · Route: which contract?", "max idf of name word in query\n+ preamble BM25\ntop-1 correct 99.6%\nkeeps ≤3 close contracts", "#e1d5e7", "#9673a6"),
    "dense":   (710, 325, 200, 70, "Dense scorer", "bge-small cosine\n(query encoded on CPU, ONNX)", "#dae8fc", "#6c8ebf"),
    "sparse":  (710, 410, 200, 70, "SPLADE scorer", "learned term weights\nsparse dot product", "#dae8fc", "#6c8ebf"),
    "rrf":     (965, 295, 190, 80, "2 · Rank: RRF fusion", "Σ 1 / (60 + rank)", "#d5e8d4", "#82b366"),
    "out":     (965, 400, 190, 100, "Top-k cited spans", "doc · char offsets · section\nR@64 81.9 · P@1 13.5\noptional: ask --llm claude", "#d5e8d4", "#82b366"),
    "eval":    (560, 507, 370, 38, "Evaluation", "char-level P@k / R@k vs gold spans · dev/test split by document", "#f8cecc", "#b85450"),
}
EDGES = [("corpus", "chunk"), ("corpus", "rindex"), ("chunk", "ctx"), ("ctx", "enc"),
         ("rindex", "route", "dashed"), ("enc", "score", "dashed"),
         ("q", "route"), ("route", "score"), ("score", "rrf"), ("rrf", "out"), ("out", "eval", "dashed")]
NOTES = [(250, 478, 300, "BM25 + IBM templates do routing only:\nremoving BM25 from ranking gave +2.7 R@8")]


def _box(key):
    return BOXES[key][:4] if key in BOXES else GROUPS[key][:4]


def _anchor(a, b):
    ax, ay, aw, ah = _box(a)
    bx, by, bw, bh = _box(b)
    acx, acy, bcx, bcy = ax + aw / 2, ay + ah / 2, bx + bw / 2, by + bh / 2
    if abs(bcy - acy) > abs(bcx - acx):  # mostly vertical: connect bottom/top edges
        return (acx, ay + ah if bcy > acy else ay), (bcx, by if bcy > acy else by + bh)
    return (ax + aw if bcx > acx else ax, acy), (bx if bcx > acx else bx + bw, bcy)


def svg() -> str:
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         'font-family="Helvetica, Arial, sans-serif">',
         '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         'orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="#444"/></marker></defs>',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    for x, y, w, h, t, fill in LANES:
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="#cccccc"/>')
        p.append(f'<text x="{x + 12}" y="{y + 22}" font-size="13" font-weight="bold" fill="#555">{escape(t)}</text>')
    for x, y, w, h, t in GROUPS.values():
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#ffffff" fill-opacity="0.6" '
                 'stroke="#6c8ebf" stroke-dasharray="4,3"/>')
        p.append(f'<text x="{x + w / 2}" y="{y + 20}" font-size="12" font-weight="bold" text-anchor="middle" '
                 f'fill="#3a5a8c">{escape(t)}</text>')
    for e in EDGES:
        (x1, y1), (x2, y2) = _anchor(e[0], e[1])
        dash = ' stroke-dasharray="5,4"' if len(e) > 2 else ""
        p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#444" stroke-width="1.4"{dash} '
                 'marker-end="url(#a)"/>')
    for x, y, w, h, title, sub, fill, stroke in BOXES.values():
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" '
                 'stroke-width="1.5"/>')
        lines = sub.split("\n")
        top = y + h / 2 - (len(lines) * 14 + 16) / 2 + 13
        p.append(f'<text x="{x + w / 2}" y="{top}" font-size="13" font-weight="bold" text-anchor="middle" '
                 f'fill="#222">{escape(title)}</text>')
        for i, line in enumerate(lines):
            p.append(f'<text x="{x + w / 2}" y="{top + 17 + 14 * i}" font-size="11" text-anchor="middle" '
                     f'fill="#444">{escape(line)}</text>')
    for x, y, w, text in NOTES:
        for i, line in enumerate(text.split("\n")):
            p.append(f'<text x="{x}" y="{y + 14 * i}" font-size="11" font-style="italic" fill="#9673a6">'
                     f'{escape(line)}</text>')
    p.append("</svg>")
    return "\n".join(p)


def drawio() -> str:
    cells, n = [], 2
    for x, y, w, h, t, fill in LANES:
        cells.append(f'<mxCell id="{n}" value="{escape(t)}" style="rounded=1;arcSize=3;fillColor={fill};'
                     'strokeColor=#cccccc;verticalAlign=top;align=left;spacingLeft=10;fontStyle=1;fontColor=#555555;" '
                     f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
                     '</mxCell>')
        n += 1
    ids = {}
    for key, (x, y, w, h, t) in GROUPS.items():
        ids[key] = f"g_{key}"
        cells.append(f'<mxCell id="g_{key}" value="{escape(t)}" style="rounded=1;dashed=1;fillColor=#ffffff;'
                     'strokeColor=#6c8ebf;verticalAlign=top;fontStyle=1;fontColor=#3a5a8c;" vertex="1" parent="1">'
                     f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    for key, (x, y, w, h, title, sub, fill, stroke) in BOXES.items():
        ids[key] = f"b_{key}"
        value = escape(f"<b>{title}</b><br><font style='font-size:11px'>{sub.replace(chr(10), '<br>')}</font>",
                       {'"': "&quot;"})
        cells.append(f'<mxCell id="b_{key}" value="{value}" style="rounded=1;whiteSpace=wrap;html=1;'
                     f'fillColor={fill};strokeColor={stroke};" vertex="1" parent="1">'
                     f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    for e in EDGES:
        dash = "dashed=1;" if len(e) > 2 else ""
        cells.append(f'<mxCell id="{n}" style="endArrow=block;html=1;{dash}strokeColor=#444444;" edge="1" '
                     f'parent="1" source="{ids[e[0]]}" target="{ids[e[1]]}"><mxGeometry relative="1" '
                     'as="geometry"/></mxCell>')
        n += 1
    for x, y, w, text in NOTES:
        cells.append(f'<mxCell id="{n}" value="{escape(text.replace(chr(10), "<br>"), {chr(34): "&quot;"})}" '
                     'style="text;html=1;fontStyle=2;fontColor=#9673a6;fontSize=11;" vertex="1" parent="1">'
                     f'<mxGeometry x="{x}" y="{y - 12}" width="{w}" height="34" as="geometry"/></mxCell>')
        n += 1
    return ('<mxfile host="app.diagrams.net"><diagram name="Blended RAG architecture">'
            f'<mxGraphModel dx="{W}" dy="{H}" grid="1" gridSize="10" pageWidth="{W}" pageHeight="{H}">'
            '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) +
            "</root></mxGraphModel></diagram></mxfile>\n")


if __name__ == "__main__":
    (OUT / "architecture.svg").write_text(svg())
    (OUT / "architecture.drawio").write_text(drawio())
    print("wrote docs/architecture.svg and docs/architecture.drawio")
