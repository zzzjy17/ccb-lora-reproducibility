from pathlib import Path


OUT = Path(__file__).with_name("fig_model_architecture_3d.svg")


def e(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, s, size=16, weight=400, fill="#1c3447", anchor="start"):
    return f'<text x="{x}" y="{y}" font-size="{size}px" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{e(s)}</text>'


def poly(points, fill, stroke="#243b4d", sw=1.8, dash=None):
    p = " ".join(f"{x},{y}" for x, y in points)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'


def line(x1, y1, x2, y2, stroke="#7a8d99", sw=2.5, marker="url(#arrow)", dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}" marker-end="{marker}"{d}/>'


def path(d, stroke="#7a8d99", sw=2.5, marker="url(#arrow)", dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}" marker-end="{marker}"{extra}/>'


def cube(x, y, w, h, depth, front, top, side, stroke="#244255", sw=2):
    # Front is the large readable face; the top and right side create a shallow
    # perspective effect while preserving true vector editability.
    return "".join(
        [
            poly([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], front, stroke, sw),
            poly([(x, y), (x + depth, y - depth * 0.62), (x + w + depth, y - depth * 0.62), (x + w, y)], top, stroke, sw),
            poly([(x + w, y), (x + w + depth, y - depth * 0.62), (x + w + depth, y + h - depth * 0.62), (x + w, y + h)], side, stroke, sw),
        ]
    )


def layer_stack(x, y, w, h, n, gap, colors, depth=18, stroke="#244255"):
    out = []
    for i in range(n - 1, -1, -1):
        ox = i * gap
        oy = -i * gap * 0.62
        front, top, side = colors[i % len(colors)]
        out.append(cube(x + ox, y + oy, w, h, depth, front, top, side, stroke))
    return "".join(out)


def label(x, y, title, lines, color="#1c3447"):
    out = [txt(x, y, title, 18, 700, color)]
    for i, s in enumerate(lines):
        out.append(txt(x, y + 23 + i * 19, s, 14, 400, "#304957"))
    return "".join(out)


def build():
    W, H = 1800, 1060
    navy = "#16374f"
    blue = "#2f73bf"
    teal = "#15938b"
    orange = "#d67825"
    purple = "#7654a8"
    gray = "#657987"
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L12,6 L0,12 z" fill="#7a8d99"/></marker>',
        '<marker id="arrow-orange" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L12,6 L0,12 z" fill="#d67825"/></marker>',
        '<filter id="shadow" x="-20%" y="-30%" width="150%" height="170%"><feDropShadow dx="0" dy="4" stdDeviation="5" flood-color="#1d3b4d" flood-opacity="0.14"/></filter>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;letter-spacing:0}.shadow{filter:url(#shadow)}</style>',
        "</defs>",
        '<rect x="18" y="18" width="1764" height="1024" rx="20" fill="#ffffff" stroke="#d7e0e6" stroke-width="1.8"/>',
        txt(W / 2, 58, "DCR-CCB-LoRA: three-dimensional model architecture", 29, 700, navy, "middle"),
        txt(W / 2, 86, "A structural view of the encoder, low-rank adaptation, calibration and controlled pseudo-label loop", 15, 400, "#6c7c85", "middle"),
        # dashed group regions
        poly([(55, 122), (1735, 122), (1735, 455), (55, 455)], "#f7fbff", "#c9ddf0", 1.5, "8 7"),
        poly([(55, 492), (1735, 492), (1735, 742), (55, 742)], "#f7fcfb", "#c8e7e0", 1.5, "8 7"),
        poly([(55, 780), (1735, 780), (1735, 1000), (55, 1000)], "#fffaf5", "#efd8bd", 1.5, "8 7"),
        txt(78, 153, "A  ENCODER AND LOW-RANK ADAPTATION", 18, 700, blue),
        txt(78, 523, "B  CALIBRATION-AWARE DYNAMIC GATING", 18, 700, teal),
        txt(78, 811, "C  CONTROLLED PSEUDO-LABEL RETRAINING", 18, 700, orange),
        # input feature planes
        cube(100, 236, 145, 112, 30, "#d8eafa", "#edf6ff", "#b3d2ed", blue),
        cube(111, 224, 145, 112, 30, "#e6f2fd", "#f5faff", "#c4def2", blue),
        label(102, 384, "Text input", ["labeled xᵢ, yᵢ", "unlabeled uⱼ", "token IDs + mask"], blue),
        # embedding / tokenizer
        line(285, 276, 355, 276),
        layer_stack(368, 250, 96, 118, 4, 14, [
            ("#dcecff", "#f1f7ff", "#b9d7f0"),
            ("#d1e6fa", "#eaf4ff", "#a8cdec"),
        ], depth=18, stroke=blue),
        label(353, 393, "Embedding", ["subword vectors", "positional encoding"], blue),
        # Transformer stack
        line(505, 276, 572, 276),
        layer_stack(592, 200, 132, 184, 6, 16, [
            ("#cce2f7", "#e8f3ff", "#9fc9eb"),
            ("#b9d8f3", "#dff0ff", "#87bbe2"),
            ("#a8cfee", "#d2e8fc", "#76afd9"),
        ], depth=22, stroke=blue),
        # attention head mini tiles
        *[
            poly(
                [
                    (629 + c * 23, 245 + r * 27),
                    (646 + c * 23, 245 + r * 27),
                    (646 + c * 23, 262 + r * 27),
                    (629 + c * 23, 262 + r * 27),
                ],
                "#f7fbff",
                blue,
                1,
            )
            for r in range(2)
            for c in range(4)
        ],
        label(585, 421, "Chinese BERT", ["multi-head self-attention", "feed-forward blocks", "shared contextual features"], blue),
        # LoRA branch above the backbone
        path("M690 197 C760 148 846 148 915 198", stroke=teal, sw=2.4, marker="url(#arrow)"),
        txt(805, 142, "trainable low-rank branch", 14, 600, teal, "middle"),
        cube(776, 179, 42, 98, 20, "#d9f2ee", "#eefbf9", "#a3d8d0", teal),
        cube(858, 179, 42, 98, 20, "#d9f2ee", "#eefbf9", "#a3d8d0", teal),
        line(818, 228, 858, 228, stroke=teal, sw=2.1),
        txt(797, 301, "A", 18, 700, teal, "middle"),
        txt(879, 301, "B", 18, 700, teal, "middle"),
        txt(838, 326, "W′ = W₀ + (α/r)BA", 15, 600, teal, "middle"),
        # feature maps / classifier head
        line(764, 276, 955, 276),
        layer_stack(978, 221, 112, 144, 4, 17, [
            ("#d8eefa", "#eef8ff", "#acd2ec"),
            ("#c9e3f6", "#e6f3ff", "#96c4e4"),
        ], depth=24, stroke=blue),
        label(975, 420, "Feature maps", ["hᵢ / hⱼ", "task-conditioned features"], blue),
        line(1155, 276, 1224, 276),
        cube(1242, 229, 125, 121, 28, "#d4e8f8", "#eef7ff", "#a4cae7", blue),
        label(1240, 392, "Classifier head", ["logits z", "K-way output"], blue),
        # calibration module
        line(1388, 276, 1450, 276),
        cube(1466, 216, 116, 145, 30, "#d9f1ed", "#effbf9", "#a6d7d0", teal),
        label(1458, 410, "Calibration", ["validation logits", "temperature T", "p = softmax(z/T)"], teal),
        # middle: dynamic gate as stacked slabs
        line(1150, 590, 1222, 590),
        cube(1240, 536, 148, 104, 28, "#ffe5cd", "#fff5eb", "#eebd91", orange),
        cube(1261, 522, 148, 104, 28, "#ffeeda", "#fff8f0", "#f2c69f", orange),
        cube(1282, 508, 148, 104, 28, "#fff2e6", "#fffaf5", "#f6d0ad", orange),
        txt(1356, 564, "DCR", 24, 700, orange, "middle"),
        txt(1356, 588, "dynamic gate", 15, 600, orange, "middle"),
        label(1238, 682, "DCR gate", ["confidence + margin", "consistency + dispersion", "class budget"], orange),
        # calibration to gate path
        path("M1526 360 L1526 458 L1350 458 L1350 508", stroke=teal, sw=2.5, dash="8 7"),
        txt(1592, 482, "calibrated confidence", 13, 400, teal, "end"),
        # unlabeled branch into gate
        cube(101, 559, 125, 92, 26, "#d9f1ed", "#effbf9", "#a4d7cf", teal),
        label(98, 689, "Unlabeled pool", ["uⱼ candidates", "same encoder"], teal),
        line(253, 600, 336, 600),
        cube(355, 548, 120, 105, 26, "#d2ece8", "#ecfaf7", "#9bd0c7", teal),
        label(352, 689, "Candidate logits", ["pⱼ, margin", "consistency checks"], teal),
        line(502, 600, 588, 600),
        cube(607, 538, 125, 115, 28, "#d2ece8", "#ecfaf7", "#9bd0c7", teal),
        label(603, 689, "Scoring", ["uncertainty", "dispersion", "class quota"], teal),
        line(759, 600, 844, 600),
        cube(861, 541, 146, 110, 28, "#ffe4cc", "#fff4e9", "#efbc8b", orange),
        label(857, 689, "Pseudo-label", ["ŷⱼ = argmax pⱼ", "weight wⱼ"], orange),
        line(1035, 600, 1222, 600),
        # bottom retraining
        cube(111, 856, 165, 96, 27, "#ffe8d2", "#fff5ec", "#edbd8b", orange),
        label(108, 980, "Merged set", ["human + accepted labels"], orange),
        line(307, 904, 386, 904, stroke=orange, marker="url(#arrow-orange)"),
        cube(405, 844, 165, 104, 27, "#ffe6cf", "#fff4e9", "#edba85", orange),
        label(402, 980, "Weighted loss", ["weighted CE", "LoRA update"], orange),
        line(601, 904, 682, 904, stroke=orange, marker="url(#arrow-orange)"),
        cube(701, 834, 165, 114, 27, "#d6eee9", "#effaf8", "#9dcec5", teal),
        label(698, 980, "Final model", ["class-aware logits", "calibrated output"], teal),
        line(897, 904, 978, 904),
        cube(998, 844, 165, 104, 27, "#e9eef2", "#f8fafb", "#bdc9d0", gray),
        label(995, 980, "Evaluation", ["Macro-F1, ECE", "Brier, audit"], gray),
        # feedback line from gate to merged set
        path("M1356 612 L1356 756 L192 756 L192 856", stroke=orange, sw=2.4, marker="url(#arrow-orange)", dash="8 7"),
        txt(780, 776, "accepted pseudo-labels with calibrated weights", 13, 600, orange, "middle"),
        # compact legend
        poly([(1210, 979), (1725, 979), (1725, 1018), (1210, 1018)], "#ffffff", "#dfe6ea", 1.2),
        txt(1230, 1004, "solid: forward data flow", 13, 400, "#526975"),
        txt(1440, 1004, "dashed: calibration / feedback path", 13, 400, "#526975"),
        "</svg>",
    ]
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    build()
