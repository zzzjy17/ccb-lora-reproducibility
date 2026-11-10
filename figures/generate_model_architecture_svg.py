from pathlib import Path


OUT = Path(__file__).with_name("fig_model_architecture.svg")


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def rect(x, y, w, h, fill, stroke="#213547", sw=2, rx=12, cls=""):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" class="{cls}"/>'
    )


def text(x, y, value, size=18, weight=400, fill="#183042", anchor="start", cls=""):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}px" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}" class="{cls}">{esc(value)}</text>'
    )


def line(x1, y1, x2, y2, stroke="#78909c", sw=2.5, marker="url(#arrow)", dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}" marker-end="{marker}"{extra}/>'


def path(d, stroke="#78909c", sw=2.5, marker="url(#arrow)", fill="none", dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" marker-end="{marker}"{extra}/>'


def block(x, y, w, h, title, lines, fill, accent, title_size=17, body_size=14):
    out = [rect(x, y, w, h, fill, stroke=accent, sw=2.4, rx=12)]
    out.append(text(x + 16, y + 28, title, title_size, 700, accent))
    for i, value in enumerate(lines):
        out.append(text(x + 16, y + 54 + i * 20, value, body_size, 400, "#243746"))
    return "".join(out)


def build_svg():
    W, H = 1680, 1030
    navy = "#17324d"
    blue = "#2f6fbd"
    teal = "#168f87"
    orange = "#d87826"
    slate = "#536b7a"
    light_blue = "#edf5ff"
    light_teal = "#eaf8f5"
    light_orange = "#fff3e8"
    light_slate = "#f2f5f7"
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L12,6 L0,12 z" fill="#78909c"/>',
        "</marker>",
        '<filter id="shadow" x="-10%" y="-10%" width="120%" height="130%"><feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#1b3445" flood-opacity="0.12"/></filter>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;letter-spacing:0} .panel{filter:url(#shadow)} .small{font-size:13px}</style>',
        "</defs>",
        rect(18, 18, W - 36, H - 36, "#ffffff", stroke="#d9e1e7", sw=1.5, rx=18),
        text(W / 2, 58, "DCR-CCB-LoRA: calibration-aware dynamic pseudo-labeling architecture", 27, 700, navy, "middle"),
        text(W / 2, 84, "Internal model structure and training data flow", 15, 400, "#667984", "middle"),
        # section labels
        rect(48, 112, 1580, 245, "#f8fbff", stroke="#d6e5f4", sw=1.5, rx=14),
        rect(48, 382, 1580, 292, "#fbfdfc", stroke="#d7ece8", sw=1.5, rx=14),
        rect(48, 699, 1580, 284, "#fffdf9", stroke="#f1dfcb", sw=1.5, rx=14),
        text(72, 143, "A  Supervised LoRA warm start", 19, 700, blue),
        text(72, 413, "B  Calibration and DCR dynamic gate", 19, 700, teal),
        text(72, 730, "C  Controlled pseudo-label retraining", 19, 700, orange),
        # top row
        block(86, 178, 190, 130, "Labeled data", ["xᵢ, yᵢ", "10% labeled split", "class-balanced sampling"], light_blue, blue),
        block(331, 178, 210, 130, "Tokenizer", ["Chinese text", "subword IDs", "attention mask"], light_blue, blue),
        block(596, 163, 250, 160, "Chinese BERT encoder", ["contextual features", "hᵢ = BERT(xᵢ)", "shared backbone"], light_blue, blue),
        block(901, 163, 238, 160, "LoRA adapters", ["low-rank updates", "W′ = W₀ + (α/r)BA", "task-specific capacity"], light_teal, teal),
        block(1200, 178, 190, 130, "Classifier", ["logits zᵢ", "K classes", "supervised loss"], light_blue, blue),
        line(276, 243, 331, 243),
        line(541, 243, 596, 243),
        line(846, 243, 901, 243),
        line(1139, 243, 1200, 243),
        # middle row inputs and calibration
        block(86, 460, 220, 148, "Unlabeled data", ["uⱼ", "unlabeled pool", "candidate examples"], light_teal, teal),
        block(350, 460, 220, 148, "Shared encoder", ["BERT + LoRA", "features hⱼ", "candidate logits"], light_teal, teal),
        block(614, 445, 235, 178, "Validation calibration", ["validation logits", "temperature scaling", "pⱼ = softmax(zⱼ/T)", "calibrated confidence"], light_teal, teal, body_size=13),
        block(901, 445, 280, 178, "DCR dynamic gate", ["confidence + margin", "consistency + dispersion", "class budget", "qⱼ = ∏ indicators"], light_orange, orange, body_size=13),
        block(1232, 460, 300, 148, "Weighted pseudo-label set", ["ŷⱼ = argmax pⱼ", "weight wⱼ = calibrated score", "accepted examples only"], light_orange, orange, body_size=13),
        line(306, 534, 350, 534),
        line(570, 534, 614, 534),
        line(849, 534, 901, 534),
        line(1181, 534, 1232, 534),
        # skip from supervised classifier to calibration
        path("M1295 308 L1295 376 L731 376 L731 445", stroke=blue, sw=2.2, dash="8 7"),
        text(1010, 367, "validation logits", 13, 400, blue, "middle"),
        # LoRA shared path from top encoder to middle encoder
        path("M721 323 L721 408 L460 408 L460 460", stroke=teal, sw=2.2, dash="8 7"),
        text(488, 402, "shared backbone + adapter weights", 13, 400, teal),
        # feedback gate loop to classifier
        path("M1382 608 L1382 662 L1220 662 L1220 308", stroke=orange, sw=2.5),
        text(1352, 653, "selected pseudo-labels", 13, 400, orange, "end"),
        # bottom row
        block(86, 776, 270, 140, "Merged training set", ["(xᵢ, yᵢ) + (uⱼ, ŷⱼ)", "human labels + accepted labels", "sample weights"], light_orange, orange, body_size=13),
        block(420, 776, 255, 140, "Weighted retraining", ["weighted cross-entropy", "LoRA parameters updated", "second-stage checkpoint"], light_orange, orange, body_size=13),
        block(742, 776, 265, 140, "Final classifier", ["calibrated logits", "class-aware prediction", "confidence audit"], light_teal, teal, body_size=13),
        block(1075, 776, 250, 140, "Evaluation", ["Macro-F1 / Accuracy", "ECE / Brier score", "seed variation"], light_slate, slate, body_size=13),
        block(1393, 776, 180, 140, "Audit", ["pseudo-label", "class budget", "error cases"], light_slate, slate, body_size=13),
        line(356, 846, 420, 846),
        line(675, 846, 742, 846),
        line(1007, 846, 1075, 846),
        line(1325, 846, 1393, 846),
        # feedback from pseudo labels into merged set
        path("M1382 608 L1382 714 L221 714 L221 776", stroke=orange, sw=2.3, dash="8 7"),
        text(800, 705, "controlled pseudo-label expansion", 13, 400, orange, "middle"),
        # legend
        rect(80, 940, 1510, 26, "#ffffff", stroke="#e5eaee", sw=1, rx=7),
        rect(101, 947, 15, 15, light_blue, stroke=blue, sw=1.5, rx=3),
        text(126, 961, "supervised backbone", 13, 400, "#435662"),
        rect(300, 947, 15, 15, light_teal, stroke=teal, sw=1.5, rx=3),
        text(325, 961, "calibration / model inference", 13, 400, "#435662"),
        rect(585, 947, 15, 15, light_orange, stroke=orange, sw=1.5, rx=3),
        text(610, 961, "pseudo-label control", 13, 400, "#435662"),
        rect(825, 947, 15, 15, light_slate, stroke=slate, sw=1.5, rx=3),
        text(850, 961, "evaluation and audit", 13, 400, "#435662"),
        line(1190, 954, 1232, 954, stroke="#78909c", sw=2.2),
        text(1244, 961, "solid: data flow   dashed: shared or feedback path", 13, 400, "#435662"),
        "</svg>",
    ]
    OUT.write_text("\n".join(svg), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    build_svg()
