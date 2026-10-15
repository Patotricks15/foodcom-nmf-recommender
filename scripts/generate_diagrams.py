"""Generate the two architecture diagrams as editable draw.io files.

Run with a dedicated venv so drawpyo does not leak into the project deps:

    python -m venv .diagram-venv && .diagram-venv/bin/pip install drawpyo
    .diagram-venv/bin/python scripts/generate_diagrams.py

Then rasterize with the draw.io exporter:

    docker run --rm -v "$PWD/docs":/data -w /data rlespinasse/drawio-export \
      -f svg -o . --output-mode relative --remove-page-suffix .
    docker run --rm -v "$PWD/docs":/data -w /data rlespinasse/drawio-export \
      -f png -o . --output-mode relative --remove-page-suffix .

    Do not pass -t/--transparent to the PNG export: it produces a
    transparent background instead of the white one the README expects.

Finally patch the exported SVGs so they carry a real white background rect
that survives GitHub's HTML sanitizer (an inline ``style="background:#fff"``
on the ``<svg>`` root is stripped on render):

    python scripts/generate_diagrams.py --patch-svg-background
"""

from __future__ import annotations

import re
from pathlib import Path

import drawpyo
from drawpyo.diagram import Edge

DOCS = Path(__file__).resolve().parents[1] / "docs"

AWS_SHAPE = (
    "sketch=0;outlineConnect=0;fontColor=#232F3E;gradientColor=none;fillColor=#{fill};"
    "strokeColor=none;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;"
    "align=center;html=1;fontSize=11;fontStyle=0;aspect=fixed;"
    "shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.{icon};"
)
BOX = (
    "rounded=1;whiteSpace=wrap;html=1;fillColor=#{fill};strokeColor=#{stroke};"
    "fontSize=11;fontColor=#232F3E;align=center;verticalAlign=middle;"
)

ORANGE, BLUE, GREEN, PURPLE = "D45B07", "4D72F3", "277116", "8C4FFF"


def aws_node(page, title, icon, fill, x, y):
    node = drawpyo.diagram.Object(page=page, value=title)
    node.apply_style_string(AWS_SHAPE.format(fill=fill, icon=icon))
    node.position = (x, y)
    node.width, node.height = 78, 78
    return node


def box(page, title, x, y, width=190, height=58, fill="FFFFFF", stroke="9AA5B1"):
    node = drawpyo.diagram.Object(page=page, value=title)
    node.apply_style_string(BOX.format(fill=fill, stroke=stroke))
    node.position = (x, y)
    node.width, node.height = width, height
    return node


def link(page, source, target, label=""):
    """Connect two nodes.

    drawpyo's Edge has no apply_style_string, so styling is passed through the
    constructor instead.
    """
    return Edge(
        page=page,
        source=source,
        target=target,
        label=label,
        fontSize=10,
        fontColor="#4A5568",
        strokeColor="#4A5568",
        endArrow="blockThin",
        rounded=1,
    )


def build_aws_architecture() -> None:
    """Offline training on the left, online serving on the right."""
    file = drawpyo.File(file_name="aws-architecture.drawio", file_path=str(DOCS))
    page = drawpyo.Page(file=file, name="AWS architecture")

    box(page, "Offline — nightly retraining", 30, 20, 450, 36, fill="FFF4E6", stroke="D45B07")
    box(
        page, "Online — request path (p99 < 50 ms)",
        560, 20, 530, 36, fill="EEF2FF", stroke="4D72F3",
    )

    kaggle = box(page, "Food.com reviews\nusers · recipes · timestamps", 30, 100, 200, 58)
    raw = aws_node(page, "S3\nraw", "bucket", GREEN, 300, 90)
    train = aws_node(page, "ECS Fargate\nNMF training", "machine_learning", ORANGE, 300, 260)
    schedule = aws_node(page, "EventBridge\nnightly 03:00", "event_time_based", PURPLE, 90, 260)
    artifacts = aws_node(page, "S3\nartifacts", "bucket", GREEN, 300, 430)
    topn = aws_node(page, "DynamoDB\ntop-N lists", "dynamodb", BLUE, 600, 430)

    client = aws_node(page, "Client", "client", BLUE, 600, 90)
    api = aws_node(page, "API Gateway", "api_gateway", PURPLE, 820, 90)
    lam = aws_node(page, "Lambda\ninference", "lambda_function", ORANGE, 820, 260)
    logs = aws_node(page, "CloudWatch\nlogs & metrics", "cloudwatch_2", PURPLE, 1010, 430)

    link(page, kaggle, raw, "ingest")
    link(page, schedule, train, "cron")
    link(page, raw, train, "read")
    link(page, train, artifacts, "W, H factors")
    link(page, artifacts, topn, "publish top-N")
    link(page, client, api, "GET /recommend")
    link(page, api, lam, "invoke")
    link(page, lam, topn, "GetItem")
    link(page, lam, logs, "emit")

    file.write()
    print(f"wrote {DOCS / 'aws-architecture.drawio'}")


def build_recommendation_architecture() -> None:
    """The modelling pipeline, from raw reviews to a filtered ranked list.

    Laid out as a single vertical spine with side branches so that no edge
    has to cross a node box.
    """
    file = drawpyo.File(file_name="recommendation-architecture.drawio", file_path=str(DOCS))
    page = drawpyo.Page(file=file, name="Recommendation architecture")

    spine_x = 420
    w, gap = 260, 118

    raw = box(
        page, "Raw Food.com reviews\nuser · recipe · rating · time",
        spine_x, 40, w, 58, fill="F7FAFC",
    )
    implicit = box(
        page,
        "Explicit + implicit views\nrating retained · rating ≥ 4 → positive",
        spine_x, 40 + gap, w, 72, fill="FFF4E6", stroke="D45B07",
    )
    core = box(
        page, "Iterative k-core filter\n≥5 interactions per user/recipe",
        spine_x, 40 + 2 * gap, w, 58, fill="FFF4E6", stroke="D45B07",
    )
    split = box(
        page, "Global temporal split\ntrain < τ₁ · val < τ₂ · test",
        spine_x, 40 + 3 * gap, w, 58, fill="FFF4E6", stroke="D45B07",
    )
    sampler = box(
        page, "Point-in-time feature build\nfit scalers on training only",
        spine_x, 40 + 4 * gap, w, 58, fill="E6FFFA", stroke="277116",
    )
    nmf = box(
        page,
        "NMF\nX ≈ WH, W,H ≥ 0\nmultiplicative updates",
        spine_x, 40 + 5 * gap, w, 86, fill="EEF2FF", stroke="4D72F3",
    )
    filter_box = box(
        page,
        "Nutrition-aware post-filter\ncalories · sodium · allergens",
        spine_x, 55 + 6 * gap, w, 72, fill="FFFAF0", stroke="D45B07",
    )
    evaluation = box(
        page,
        "Evaluation\nNDCG@10 · Recall@10 · MAP@10 · Coverage\nfilter-impact delta",
        spine_x, 70 + 7 * gap, w, 86, fill="FFF5F5", stroke="C53030",
    )

    # Side branches, offset far enough that their edges stay clear of the spine.
    features = box(
        page,
        "Feature registry\nconf/features.yaml\nstrict cut-off (<)",
        spine_x - 330, 40 + 4 * gap - 6, 240, 72, fill="E6FFFA", stroke="277116",
    )
    baselines = box(
        page, "Baselines\nPopularity · shrunk Item-kNN",
        spine_x - 330, 40 + 5 * gap + 14, 240, 58, fill="F7FAFC",
    )
    serving = box(
        page, "Top-N publication\nDynamoDB · parity-tested",
        spine_x + 330, 70 + 7 * gap - 14, 240, 58, fill="EEF2FF", stroke="4D72F3",
    )

    link(page, raw, implicit)
    link(page, implicit, core)
    link(page, core, split)
    link(page, split, sampler)
    link(page, sampler, nmf, "sparse matrix")
    link(page, nmf, filter_box, "ranked candidates")
    link(page, filter_box, evaluation)
    link(page, core, features, "train only")
    link(page, features, sampler)
    link(page, baselines, nmf)
    link(page, filter_box, serving, "top-N")

    file.write()
    print(f"wrote {DOCS / 'recommendation-architecture.drawio'}")


def patch_svg_white_background(svg_path: Path) -> None:
    """Insert a literal white background rect into an exported SVG.

    GitHub's HTML sanitizer strips inline ``style="background: ..."`` from
    the ``<svg>`` root, so a diagram exported with a transparent canvas
    renders invisible against GitHub's dark theme. The fix is a real
    ``<rect>`` painted white, inserted as the first child of the outermost
    ``<g>`` (right after ``<defs/>``), which sanitizers do not strip because
    it is ordinary drawable content rather than a style attribute.

    Idempotent: running this twice does not insert a second rect.

    Args:
        svg_path: Path to an SVG exported by ``rlespinasse/drawio-export``.
    """
    text = svg_path.read_text()
    width_match = re.search(r'width="(\d+)px"', text)
    height_match = re.search(r'height="(\d+)px"', text)
    if not (width_match and height_match):
        raise ValueError(f"{svg_path} has no pixel width/height to size the background rect.")
    width, height = width_match.group(1), height_match.group(1)
    rect = f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>'

    if rect in text:
        return  # already patched

    anchor = "<defs/><g>"
    if anchor not in text:
        raise ValueError(f"{svg_path} does not match the expected <defs/><g> export shape.")

    patched = text.replace(anchor, anchor + rect, 1)
    svg_path.write_text(patched)
    print(f"patched white background into {svg_path}")


if __name__ == "__main__":
    import sys

    DOCS.mkdir(exist_ok=True)
    if "--patch-svg-background" in sys.argv:
        for svg in sorted(DOCS.glob("*.svg")):
            patch_svg_white_background(svg)
    else:
        build_aws_architecture()
        build_recommendation_architecture()
