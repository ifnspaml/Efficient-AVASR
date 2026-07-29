#!/usr/bin/env python3
"""Build the Chapter 3 checkpoint-failure analysis from the Word template."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw, ImageFont


REFERENCE = Path(
    "/home/zhengyangli/.codex/plugins/cache/openai-curated-remote/"
    "openai-templates/0.1.0/skills/"
    "artifact-template-experiment-analysis/assets/reference.docx"
)
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
}
W = f"{{{NS['w']}}}"
ET.register_namespace("w", NS["w"])
ET.register_namespace(
    "w14", "http://schemas.microsoft.com/office/word/2010/wordml"
)
ET.register_namespace(
    "wp14", "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_paragraph_text(paragraph: ET.Element, text: str) -> None:
    text_nodes = paragraph.findall(".//w:t", NS)
    if text_nodes:
        text_nodes[0].text = text
        for node in text_nodes[1:]:
            node.text = ""
        return
    run = ET.SubElement(paragraph, f"{W}r")
    node = ET.SubElement(run, f"{W}t")
    if text.startswith(" ") or text.endswith(" "):
        node.set(
            "{http://www.w3.org/XML/1998/namespace}space", "preserve"
        )
    node.text = text


def _set_cell_text(cell: ET.Element, text: str) -> None:
    paragraphs = cell.findall("./w:p", NS)
    if not paragraphs:
        paragraphs = [ET.SubElement(cell, f"{W}p")]
    _set_paragraph_text(paragraphs[0], text)
    for paragraph in paragraphs[1:]:
        _set_paragraph_text(paragraph, "")


def _set_table(table: ET.Element, rows: list[list[str]]) -> None:
    table_rows = table.findall("./w:tr", NS)
    if len(rows) != len(table_rows):
        raise ValueError(
            f"table row mismatch: template={len(table_rows)}, data={len(rows)}"
        )
    for row, values in zip(table_rows, rows):
        cells = row.findall("./w:tc", NS)
        if len(values) != len(cells):
            raise ValueError(
                f"table column mismatch: template={len(cells)}, data={len(values)}"
            )
        for cell, value in zip(cells, values):
            _set_cell_text(cell, value)


def _run_map(audit: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {run["experiment"]: run for run in audit["runs"]}


def _value(run: Dict[str, Any], condition: str) -> float:
    return float(run["wer"][condition])


def _f(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _report_content(audit: Dict[str, Any]) -> tuple[dict[int, str], list[list[list[str]]]]:
    runs = _run_map(audit)
    b1 = runs["b1_t2_teacher_init"]
    b2 = runs["b2_c1_t2_random_sequence"]
    c3a = runs["c3a_t12_historical_heads"]
    c3b = runs["c3b_t12_layer_to_layer"]
    clean_gain = _value(b2, "clean") - _value(c3a, "clean")
    clean_relative = clean_gain / _value(b2, "clean") * 100
    c3a_drop = (
        c3a["fine_tuning_validation"]["best"]["accuracy"]
        - c3a["fine_tuning_validation"]["final"]["accuracy"]
    )
    c3b_drop = (
        c3b["fine_tuning_validation"]["best"]["accuracy"]
        - c3b["fine_tuning_validation"]["final"]["accuracy"]
    )

    paragraphs = {
        0: "Experiment Report",
        6: "Chapter 3 Distillation-Only",
        7: "Checkpoint & Fine-Tuning Failure Analysis",
        23: "Codex · Efficient-AVASR",
        24: "July 29, 2026",
        30: (
            "This record investigates why B2, C3a, and C3b produced worse "
            "validation WER than B1. It tests the suspected failure modes at "
            "the three critical boundaries: teacher-front-end initialization, "
            "distillation checkpoint export, and exported-backbone loading "
            "into the unchanged 60,000-update seq2seq fine-tuning pipeline."
        ),
        31: (
            "The analysis combines immutable manifests, Hydra configs, raw "
            "checkpoint tensors, initialization reports, Fairseq learning "
            "curves, and WER artifacts. The machine-readable checkpoint audit "
            "is retained beside this report for replication."
        ),
        34: (
            "B1 (two-block, D=768) uses teacher-derived sequence and frontend "
            "initialization. B2 uses the same architecture with a random "
            "sequence model. C3a and C3b are compact 12-block D=384 students "
            "with random sequence initialization; they differ only in "
            "historical final-output heads versus layer-to-layer matching."
        ),
        35: (
            "The immediate decision is whether the current C-series WER can "
            "support C→D architecture selection. This is reversible: no "
            "test-set selection has been performed, and reruns remain isolated "
            "under exp/chapter3_distill_only."
        ),
        40: (
            "This is a retrospective implementation audit, not a new causal "
            "training experiment. All four completed seed-1337 runs use 75,000 "
            "encoder updates, the same LRS3 433 h data, targets 0/4/8/12, and "
            "the existing 60,000-update downstream pipeline. Tensor mappings "
            "were compared without constructing replacement models."
        ),
        41: (
            "The audit requires exact equality from distillation checkpoint to "
            "export. For the pre-unfreeze fine-tuning checkpoint it accepts "
            "only FP16 round-trip differences (absolute and relative tolerance "
            "0.001), separates BatchNorm buffers, and independently detects "
            "untracked interface projections."
        ),
        44: (
            "If the poor WER is caused by missing ResNet or distilled weights, "
            "then the exported and pre-unfreeze fine-tuning tensors will be "
            "absent or dissimilar. If the handoff is correct, those tensors "
            "will match and the failure should instead appear in initialization "
            "policy, interface adaptation, or downstream optimization."
        ),
        50: (
            "PASS — all student tensor names and shapes map from the 75k "
            "distillation checkpoint to the native export."
        ),
        51: (
            "PASS — all mapped distillation→export tensors are bitwise exact "
            "for B1, B2, C3a, and C3b."
        ),
        52: (
            "PASS — B2/C3 pre-unfreeze backbone parameters match their exports "
            "within FP16 round-trip tolerance; no parameter is outside tolerance."
        ),
        53: (
            "PASS WITH CAVEAT — C3 initialization reports copy all 137 visual "
            "ResNet tensors. Eight dimension-dependent audio/video/fusion "
            "projection tensors are correctly skipped and randomly initialized."
        ),
        54: (
            "FAIL — C3 fine-tuning contains a random 384→768 encoder projection "
            "introduced after the manifest commit, and all 60 ResNet BatchNorm "
            "running buffers move during the nominally frozen phase."
        ),
        57: (
            "Four completed runs and three validation conditions were analyzed. "
            "Each encoder reached 75,000 updates and each fine-tuning job reached "
            "60,000. The evidence is sufficient to diagnose the handoff and "
            "optimization path, but not to rank architectures scientifically "
            "because C3 stage provenance is mixed and only one seed is present."
        ),
        61: (
            f"C3a clean WER ({_f(_value(c3a, 'clean'))}) improves on the "
            f"controlled random-init T2 baseline B2 ({_f(_value(b2, 'clean'))}) "
            f"by {_f(clean_gain)} absolute / {_f(clean_relative, 1)}% relative. "
            "It does not beat B1 because B1 is a different initialization "
            "treatment, not the C-series control. C3 results are nevertheless "
            "invalid for final selection until the downstream interface and "
            "source-provenance issues are removed."
        ),
        65: (
            "Checkpoint integrity passes. Optimization stability fails for "
            "every random-sequence run: their best checkpoints occur near "
            "update 11.6k, before the 48k encoder unfreeze, and validation "
            f"accuracy later falls by 19.16 points (B2), {_f(c3a_drop)} "
            f"(C3a), and {_f(c3b_drop)} (C3b). Tail gradient norms are "
            "1,737–3,144 versus 19.5 for B1."
        ),
        67: (
            "The only prespecified scientific comparison represented here is "
            "C3a/C3b versus B2 under the common random-sequence policy. B1 is "
            "retained as an initialization reference, not as the valid control "
            "for depth."
        ),
        69: (
            "C3a is best on clean WER among random-init runs; C3b is best on "
            "0 dB babble. Both are worse than B2 on 0 dB competing speech. "
            "These directional patterns are confounded by the C3-only random "
            "bridge and unstable post-unfreeze training."
        ),
        72: (
            "Concurrent source change: C3 encoder training began at recorded "
            "commit 060a921, while fine-tuning ran after f417cda added the "
            "384→768 seq2seq projection."
        ),
        73: (
            "The manifests retain 060a921 as the implementation commit for all "
            "stages. C3 checkpoint keys prove that post-060a source executed, so "
            "the immutable provenance is incomplete for those runs."
        ),
        74: (
            "No update-zero downstream checkpoint exists. Loading is established "
            "by exact export provenance plus high-precision tensor agreement at "
            "the pre-unfreeze best checkpoints, allowing only FP16 conversion."
        ),
        75: (
            "The evidence covers one seed and validation data only. It does not "
            "support test-set conclusions or variance estimates."
        ),
        76: (
            "The unchanged downstream freeze is gradient-only. BatchNorm remains "
            "in training mode, so visual running statistics change even while "
            "the encoder is under no_grad."
        ),
        77: (
            "Consequently, current C3 WER should not be generalized to the "
            "12-block architecture or distillation objective. It is evidence "
            "about a mixed implementation and unstable fine-tuning path."
        ),
        80: (
            "The suspected ResNet/export/load failures are rejected. The visual "
            "trunk was copied, the deployed checkpoint exactly contains the "
            "distilled student, and fine-tuning loaded it. The observed deficit "
            "is explained by a strong B1 teacher-initialization advantage, an "
            "untrained C3-only interface bridge, mutable BatchNorm state during "
            "freeze, and severe post-unfreeze optimization collapse."
        ),
        81: (
            "Decision posture: hold C→D selection and do not spend compute on D/E "
            "from these artifacts. First make the 384→768 interface part of the "
            "isolated, trained, exported student (or explicitly change and "
            "control decoder width), lock source per stage, and run a short "
            "fine-tuning stability diagnostic before repeating C3."
        ),
        87: (
            "1. Preserve the current runs as failed diagnostic artifacts; do not "
            "overwrite or relabel them."
        ),
        88: (
            "2. Train and export any required narrow-to-decoder adapter so no "
            "new random frozen layer appears only at downstream fine-tuning."
        ),
        89: (
            "3. Add update-zero and pre-unfreeze checkpoint assertions: exported "
            "backbone parameters must be close after FP16 conversion, and every "
            "extra encoder key must be declared in the manifest."
        ),
        90: (
            "4. Run a short controlled diagnostic around update 48k, recording "
            "gradient norms, loss scale, BatchNorm policy, and encoder/projection "
            "learning rates before committing to full reruns."
        ),
        91: (
            "5. Rerun B2/C3 under one committed source tree. Use B2—not B1—as "
            "the C-series depth control; retain B1 solely for initialization."
        ),
        94: (
            "Clean WER: validation word error rate from the existing clean "
            "decoder output; lower is better."
        ),
        95: (
            "Noisy WER: validation WER under 0 dB babble or competing speech "
            "using the recorded evaluation artifacts."
        ),
        96: (
            "Checkpoint handoff exactness: torch.equal for every mapped "
            "distillation-student and export tensor."
        ),
        97: (
            "Fine-tuning load closeness: allclose at absolute/relative 0.001 "
            "for non-buffer backbone tensors before encoder unfreeze."
        ),
        98: (
            "Optimization stability: best/final validation accuracy, checkpoint "
            "update, tail gradient norm, and FP16 loss scale."
        ),
        101: (
            "Software: scripts/distill_only/audit_checkpoint_handoffs.py with "
            "raw PyTorch state dictionaries; report generated from the supplied "
            "Experiment Analysis Word template."
        ),
        102: (
            "No interim stopping or test read was used. Existing checkpoint_best "
            "selection follows validation accuracy in the unchanged pipeline."
        ),
        103: (
            "Automated tests cover prefix mapping, independent parameter/buffer "
            "mismatch detection, FP16 tolerance, projection detection, log "
            "parsing, and per-stage clean-commit enforcement."
        ),
        104: (
            "Primary evidence: manifest.v1.json, export metadata, profile.json, "
            "encoder/finetune logs, checkpoint_last.pt, checkpoint_best.pt, and "
            "checkpoint_handoff_audit.v1.json for the four seed-1337 runs."
        ),
    }

    tables = [
        [
            ["Version", "1.0"],
            ["Prepared By", "Codex · Efficient-AVASR"],
            ["Reviewers", "Chapter 3 scientific owner (pending)"],
            ["Date Prepared", "July 29, 2026"],
            ["Reporting Window", "July 25–29, 2026"],
            ["Status", "Diagnostic complete · rerun required"],
        ],
        [
            ["Field", "Details"],
            ["Experiment Name", "Chapter 3 distillation-only checkpoint audit"],
            ["Experiment Key", "B1 / B2=C1 / C3a / C3b · seed 1337"],
            ["Owner Team", "Efficient-AVASR"],
            ["Scientific Owner", "Chapter 3 experiment review"],
            ["Pipeline Surface", "75k encoder distillation → export → 60k seq2seq fine-tuning"],
            ["Primary Objective", "Explain WER gap and verify all weight handoffs"],
            ["Reference", "B1: T2 D768 teacher sequence + teacher frontend"],
            ["Controlled Depth Baseline", "B2=C1: T2 D768 random sequence + teacher frontend"],
            ["Treatments", "C3a/C3b: T12 D384 random sequence; two objectives"],
            ["Unit of Analysis", "Experiment × seed manifest"],
            ["Dataset", "LRS3 433 h; validation-only selection"],
            ["Excluded Evidence", "Test WER and unlaunched D/E rows"],
            ["Start Date", "July 25, 2026"],
            ["End Date", "July 29, 2026"],
            ["Planned Runtime", "75k encoder + 60k fine-tuning per run"],
            ["Actual Runtime", "All four reached planned update counts"],
        ],
        [
            ["Metric", "Target / Rule"],
            ["Primary Integrity Metric", "All distillation→export tensors bitwise exact"],
            ["Load Guardrail", "All pre-unfreeze backbone parameters close at 1e-3"],
            ["Frontend Guardrail", "All shape-compatible visual ResNet tensors copied"],
            ["Optimization Guardrail", "No major accuracy collapse after encoder unfreeze"],
            ["Decision Rule", "Hold C→D if provenance, interface, or stability guardrail fails"],
        ],
        [
            ["Measure", "B1", "B2 / C3 set"],
            ["Completed models / seed", "1 / 1337", "3 / 1337 each"],
            ["Encoder updates", "75,000", "75,000 each"],
            ["Fine-tuning updates", "60,000", "60,000 each"],
            ["Validation conditions", "Clean + two 0 dB conditions", "Same"],
            ["Data suitability", "Valid initialization reference", "Diagnostic only; C3 provenance mixed"],
        ],
        [
            ["Validation WER ↓", "B1", "B2=C1", "C3a", "C3b"],
            ["Clean", _f(_value(b1, "clean")), _f(_value(b2, "clean")), _f(_value(c3a, "clean")), _f(_value(c3b, "clean"))],
            ["Babble 0 dB", _f(_value(b1, "babble_0db")), _f(_value(b2, "babble_0db")), _f(_value(c3a, "babble_0db")), _f(_value(c3b, "babble_0db"))],
            ["Competing speech 0 dB", _f(_value(b1, "speech_0db")), _f(_value(b2, "speech_0db")), _f(_value(c3a, "speech_0db")), _f(_value(c3b, "speech_0db"))],
            ["Deployed parameters", "31.74 M", "31.74 M", "29.47 M", "29.47 M"],
        ],
        [
            ["Checkpoint Diagnostic", "B1", "B2=C1", "C3a", "C3b"],
            ["75k student → export", "Exact", "Exact", "Exact", "Exact"],
            ["Export → pre-unfreeze model", "N/A: best at 60k", "Close; FP16 only", "Close; FP16 only", "Close; FP16 only"],
            ["Visual ResNet copied", "137 / 137", "137 / 137", "137 / 137", "137 / 137"],
            ["Extra 384→768 projection", "No", "No", "Yes; random", "Yes; random"],
        ],
        [
            ["Run / Initialization", "Final distill loss", "Best FT accuracy @ update", "Final FT accuracy", "Interpretation"],
            ["B1 · teacher sequence", "4.9966", "90.474 @ 60,000", "90.474", "Stable; tail gnorm 19.5"],
            ["B2 · random sequence", "4.1424", "74.802 @ 11,647", "55.643", "Collapses after unfreeze; gnorm 1,737"],
            ["C3a · random / heads", "4.4311", "79.077 @ 11,646", "47.707", "Random bridge + collapse; gnorm 2,003"],
            ["C3b · random / L2L", "3.8796", "78.526 @ 11,646", "52.941", "Random bridge + collapse; gnorm 3,144"],
            ["Controlled conclusion", "Objectives differ for C3b", "C3a/C3b > B2 pre-unfreeze", "All random runs unstable", "Do not select D from current artifacts"],
        ],
        [
            ["Item", "Decision"],
            ["Final Decision", "Hold C→D selection; rerun corrected B2/C3"],
            ["Decision Date", "July 29, 2026"],
            ["Approvers", "Scientific owner approval pending"],
            ["Rerun Type", "New immutable run directories; never overwrite current artifacts"],
            ["Stop Trigger", "Any stage commit drift, undeclared encoder key, or failed handoff tolerance"],
            ["Follow Up", "Trained/exported interface adapter and update-48k stability smoke test"],
        ],
    ]
    return paragraphs, tables


def build_docx(audit: Dict[str, Any], output: Path) -> Dict[str, Any]:
    reference_before = _sha256(REFERENCE)
    paragraphs_text, table_rows = _report_content(audit)
    with ZipFile(REFERENCE) as source:
        document_xml = source.read("word/document.xml")
        root = ET.fromstring(document_xml)
        body = root.find(".//w:body", NS)
        if body is None:
            raise ValueError("template document has no body")
        paragraphs = body.findall("./w:p", NS)
        tables = body.findall("./w:tbl", NS)
        if len(paragraphs) != 106 or len(tables) != 8:
            raise ValueError(
                "unexpected template structure: "
                f"{len(paragraphs)} paragraphs, {len(tables)} tables"
            )
        for index, text in paragraphs_text.items():
            _set_paragraph_text(paragraphs[index], text)
        for table, rows in zip(tables, table_rows):
            _set_table(table, rows)

        updated_xml = ET.tostring(
            root, encoding="utf-8", xml_declaration=True
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)
        )
        os.close(descriptor)
        try:
            with ZipFile(temporary, "w", ZIP_DEFLATED) as destination:
                for item in source.infolist():
                    payload = (
                        updated_xml
                        if item.filename == "word/document.xml"
                        else source.read(item.filename)
                    )
                    destination.writestr(item, payload)
            os.replace(temporary, output)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    with ZipFile(output) as result:
        corrupt_member = result.testzip()
        parsed = ET.fromstring(result.read("word/document.xml"))
        rendered_text = "".join(
            node.text or "" for node in parsed.findall(".//w:t", NS)
        )
    reference_after = _sha256(REFERENCE)
    return {
        "schema_version": "chapter3-experiment-analysis-document/v1",
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "reference": str(REFERENCE),
        "reference_sha256_before": reference_before,
        "reference_sha256_after": reference_after,
        "reference_unchanged": reference_before == reference_after,
        "zip_test": corrupt_member,
        "document_xml_well_formed": parsed is not None,
        "template_placeholder_fragments_remaining": sum(
            rendered_text.count(fragment)
            for fragment in ("[Describe", "[Experiment", "[Metric", "[Variant")
        ),
        "paragraph_count": len(parsed.findall(".//w:body/w:p", NS)),
        "table_count": len(parsed.findall(".//w:body/w:tbl", NS)),
        "source_audit_schema": audit.get("schema_version"),
    }


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size=size)


def _wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    width: int,
    font: ImageFont.FreeTypeFont,
    fill: str,
    spacing: int = 12,
) -> int:
    lines = textwrap.wrap(text, width=width)
    draw.multiline_text(xy, "\n".join(lines), font=font, fill=fill, spacing=spacing)
    box = draw.multiline_textbbox(
        xy, "\n".join(lines), font=font, spacing=spacing
    )
    return box[3]


def build_previews(audit: Dict[str, Any], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    green = "#397a49"
    dark = "#202820"
    grey = "#657065"

    cover = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(cover)
    draw.text((95, 105), "Experiment Report", font=_font(34, True), fill=green)
    draw.rectangle((95, 425, 1145, 433), fill=green)
    draw.text(
        (95, 500), "Chapter 3 Distillation-Only", font=_font(60, True), fill=dark
    )
    _wrapped(
        draw,
        (95, 590),
        "Checkpoint & Fine-Tuning Failure Analysis",
        width=28,
        font=_font(54, True),
        fill=green,
        spacing=18,
    )
    draw.text((95, 1455), "Codex · Efficient-AVASR", font=_font(27, True), fill=dark)
    draw.text((95, 1505), "July 29, 2026", font=_font(25), fill=grey)
    draw.rectangle((95, 1652, 1145, 1657), fill=green)
    cover_path = output_dir / "preview_cover.png"
    cover.save(cover_path)

    findings = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(findings)
    draw.rectangle((0, 0, 1240, 155), fill=green)
    draw.text((80, 48), "Executive Findings", font=_font(46, True), fill="white")
    items = (
        ("PASS", "Visual ResNet copied", "137 / 137 trunk tensors in both C3 runs."),
        ("PASS", "Distillation export", "Every mapped tensor is bitwise exact in all four runs."),
        ("PASS", "Fine-tuning load", "B2/C3 backbone parameters match within FP16 tolerance."),
        ("FAIL", "C3 interface", "A random frozen 384→768 bridge appears only downstream."),
        ("FAIL", "Optimization", "Random-init runs peak at ~11.6k and collapse after 48k unfreeze."),
        ("HOLD", "Scientific selection", "C3 beats controlled B2 clean WER, but current C3 is confounded."),
    )
    y = 220
    for status, title, detail in items:
        color = green if status == "PASS" else "#9b3d36"
        if status == "HOLD":
            color = "#a36f17"
        draw.rounded_rectangle((75, y, 1165, y + 190), radius=18, fill="#f4f7f4")
        draw.rounded_rectangle((95, y + 30, 250, y + 85), radius=12, fill=color)
        draw.text((118, y + 42), status, font=_font(22, True), fill="white")
        draw.text((285, y + 30), title, font=_font(31, True), fill=dark)
        _wrapped(
            draw,
            (285, y + 86),
            detail,
            width=52,
            font=_font(24),
            fill=grey,
            spacing=8,
        )
        y += 225
    findings_path = output_dir / "preview_findings.png"
    findings.save(findings_path)
    return [str(cover_path.resolve()), str(findings_path.resolve())]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    metadata = build_docx(audit, args.output)
    metadata["previews"] = build_previews(audit, args.preview_dir)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
