#!/usr/bin/env python3
"""Build a deterministic instruction-stratified image-edit evaluation set.

The retry input manifest does not contain an edit-task label.  This script
assigns one mutually exclusive primary task from a versioned, auditable regex
taxonomy, filters to cases with an evaluation ground truth, and samples the
same number of unique cases per sufficiently populated task.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable


TAXONOMY_VERSION = "retry_instruction_primary_v3"
DEFAULT_SOURCE_S3 = (
    "s3://core-cn-oss.canva.com/usr/grantzhang/image-edit-v4/"
    "retry_failed6304_r1/"
)


@dataclass(frozen=True)
class TaskRule:
    name: str
    description: str
    patterns: tuple[str, ...]


# Rules are ordered from specific to general.  A sample receives exactly one
# primary task, using the first matching rule.  This makes the 50-per-task
# sample count equal to the number of unique evaluation cases.
TASK_RULES: tuple[TaskRule, ...] = (
    TaskRule(
        "typography_or_text_style",
        "Edit font, typography, text color, text texture, or text styling; mixed text-content edits keep this primary task.",
        (
            r"\b(?:font|typeface|typograph|text style|letter style|text color|letter color|text texture|letter texture)\w*\b",
            r"\b(?:bold|italic|serif|sans[- ]serif|outlined text|3d text|neon text|metallic text)\b",
        ),
    ),
    TaskRule(
        "text_content_or_translation",
        "Change, restore, remove, add, or translate visible textual content.",
        (
            r"\btranslat(?:e|ion)\b",
            r"\b(?:change|replace|restore|revert|remove|delete|add|insert|write) (?:the |all |any )?(?:visible )?(?:text|word|words|wording|lettering|caption|headline|title|subtitle|label)s?\b",
            r"\b(?:text|word|words|wording|lettering|caption|headline|title|subtitle|label)s?\b.{0,80}\b(?:to read|so (?:that )?it reads|from .{1,40} to|back to|translate|spell)\b",
            r"\b(?:read|reads|say|says|spell|spells)\b.{0,80}['\"“][^'\"”]{1,100}['\"”]",
        ),
    ),
    TaskRule(
        "crop_or_zoom_in",
        "Crop, tighten framing, or zoom in to a closer composition.",
        (
            r"\bcrop(?:ped|ping)?\b",
            r"\bzoom in\b",
            r"\b(?:crop|zoom|frame|reframe|shot|view)\w*\b.{0,50}\bclose[- ]up\b",
            r"\btight(?:er|ly)? (?:crop|framing|frame)\b",
        ),
    ),
    TaskRule(
        "outpaint_or_zoom_out",
        "Uncrop, outpaint, zoom out, or expand the visible canvas/framing.",
        (
            r"\b(?:uncrop|outpaint|zoom out)\b",
            r"\b(?:expand|extend|widen)\b.*\b(?:canvas|frame|framing|image|view|background)\b",
            r"\bshow (?:more|the full|the entire)\b.*\b(?:scene|body|object|building|background|surroundings)\b",
        ),
    ),
    TaskRule(
        "style_or_medium_transfer",
        "Convert between photo, painting, illustration, sketch, vector, 3D, or another visual style.",
        (
            r"\b(?:convert|transform|restyle|redesign|render|turn|change|restore)\w*\b.{0,110}\b(?:style|stylized|realistic photograph|digital photograph|oil painting|watercolor|impressionist|cubist|pop[- ]art|charcoal|pencil sketch|line drawing|illustration|cartoon|anime|flat design|vector|3d render|photorealistic)\b",
            r"\b(?:oil painting|watercolor|impressionist|cubist|pop[- ]art|charcoal|pencil sketch|line drawing|illustration|cartoon|anime|flat design|vector|3d render)\b.{0,110}\b(?:into|to|back to|style|photo|photograph|realistic|convert|transform|restore)\b",
        ),
    ),
    TaskRule(
        "color_temperature_or_photo_filter",
        "Change white balance, color cast, vintage/film filter, grain, or global photo treatment.",
        (
            r"\b(?:color temperature|white balance|color cast|warm cast|cool cast|color grade|colour grade|color grading|colour grading|sepia|vintage|retro film|film filter|film grain|light leak|chromatic aberration|color fringing|duotone|black and white|grayscale|monochrome)\b",
        ),
    ),
    TaskRule(
        "blur_focus_or_sharpness",
        "Add/remove blur or change focus and sharpness.",
        (
            r"\b(?:blur|blurry|blurred|bokeh|defocus|refocus|focus|sharp|sharpen|soft focus|depth of field|motion blur)\w*\b",
        ),
    ),
    TaskRule(
        "perspective_or_viewpoint",
        "Change camera angle, perspective, viewpoint, or view direction.",
        (
            r"\b(?:perspective|viewpoint|camera angle|camera view|viewing angle|angle of view)\b",
            r"\b(?:top[- ]down|bird['’]?s[- ]eye|low[- ]angle|high[- ]angle|front[- ]facing|side[- ]profile|profile view|straight[- ]on|head-on|overhead view)\b",
            r"\b(?:look(?:ing)?|view(?:ed|ing)?)\b.*\b(?:from above|from below|from the side|directly down|directly at)\b",
        ),
    ),
    TaskRule(
        "rotation_flip_or_mirror",
        "Rotate, flip, mirror, tilt, or straighten an object/image.",
        (
            r"\b(?:rotate|rotation|flip|mirror|tilt|straighten|upright)\b",
        ),
    ),
    TaskRule(
        "composition_or_layout",
        "Redesign or rearrange a multi-element graphic/layout/composition.",
        (
            r"\b(?:redesign|reformat|rearrange|reorganize)\w*\b.{0,90}\b(?:layout|composition|grid|columns?|panels?|slide design|poster design)\b",
            r"\b(?:restore|change)\w*\b.{0,25}\b(?:the |its |original )?(?:layout|composition|grid|columns?|panels?)\b",
            r"\b(?:layout|composition|grid|columns?|panels?)\b.{0,90}\b(?:rearrange|reorganize|redesign|reformat|stack|distribute)\w*\b",
        ),
    ),
    TaskRule(
        "move_or_reposition",
        "Move, shift, place, align, or spatially rearrange existing content.",
        (
            r"\b(?:move|moved|reposition|shift|place|relocate|align|rearrange|reorganize)\w*\b",
            r"\b(?:swap|switch)\b.{0,30}\b(?:position|positions|places|sides)\b",
        ),
    ),
    TaskRule(
        "resize_or_scale",
        "Increase/decrease object size, scale, width, or height.",
        (
            r"\b(?:resize|rescale|scale up|scale down|enlarge|shrink|smaller|larger|bigger|reduce the size|increase the size|decrease the size)\b",
            r"\b(?:reduce|decrease|increase|change|adjust|make)\b.{0,60}\b(?:size|scale|width|height|smaller|larger|bigger|wider|narrower|shorter|longer)\b",
            r"\b(?:widen|narrow|lengthen|shorten)\b.*\b(?:object|body|waist|arm|leg|neck|shape|width|height)\b",
        ),
    ),
    TaskRule(
        "facial_expression",
        "Change facial expression, gaze, eye state, or mouth expression.",
        (
            r"\b(?:change|modify|adjust|restore|soften)\w*\b.{0,70}\b(?:facial expression|expression|gaze|smile|frown|eyes?|mouth)\b",
            r"\b(?:facial expression|expression|gaze)\b.{0,70}\b(?:change|modify|adjust|restore|soften|from|to)\b",
            r"\bmake\b.{0,50}\b(?:person|man|woman|girl|boy|child|face|subject)\b.{0,60}\b(?:smile|frown|look|expression|happy|sad|angry|surprised|shocked)\b",
        ),
    ),
    TaskRule(
        "pose_gesture_or_articulation",
        "Change a person/animal pose, gesture, limb state, or articulated-object state.",
        (
            r"\b(?:change|modify|adjust|restore|make)\w*\b.{0,80}\b(?:pose|gesture|posture|stance|perched|wings? (?:folded|raised|lowered|spread|open|closed)|arms? (?:raised|lowered|crossed|outstretched|open|closed)|legs? (?:raised|lowered|crossed|spread|open|closed))\b",
            r"\b(?:pose|gesture|posture|stance)\b.{0,70}\b(?:change|modify|adjust|restore|from|to|so)\b",
            r"\b(?:open|close|closed|shut|zip up|unzip)\b.*\b(?:door|drawer|lid|jacket|mouth|eyes|wings)\b",
        ),
    ),
    TaskRule(
        "weather_or_atmosphere",
        "Change rain, snow, fog, mist, haze, clouds, or other weather/atmosphere.",
        (
            r"\b(?:weather|rain|rainy|snow|snowy|storm|stormy|fog|foggy|mist|misty|haze|hazy|clouds?|cloudy|sunny day|clear sky|overcast)\b",
        ),
    ),
    TaskRule(
        "season_or_time_of_day",
        "Change season or time of day.",
        (
            r"\b(?:season|spring|summer|autumn|fall season|winter|daytime|nighttime|night sky|sunrise|sunset|dawn|dusk|morning|evening)\b",
        ),
    ),
    TaskRule(
        "lighting_or_exposure",
        "Change lighting, exposure, brightness, contrast, glare, or illumination.",
        (
            r"\b(?:lighting|illumination|exposure|brightness|brighten|darken|dim|contrast|specular highlights?|bright highlights?|glare|lens flare|light rays?|sunbeam|spotlight|backlit|underexposed|overexposed)\b",
        ),
    ),
    TaskRule(
        "material_or_texture",
        "Change physical material, surface texture, finish, wetness, or fabric appearance.",
        (
            r"\b(?:change|convert|replace|restore|remove|make|apply|give)\w*\b.{0,45}\b(?:material|texture|surface finish|fabric|leather|paper texture|canvas texture|brushstrokes?|woven|knit|glossy|matte|wet-looking|dry appearance)\b",
            r"\b(?:material|texture|surface finish|fabric|leather|paper texture|canvas texture|brushstrokes?|woven|knit|glossy|matte|wet-looking|dry appearance)\b.{0,60}\b(?:to|into|from|back to|look|appearance|remove|restore|replace|change)\b",
            r"\b(?:change|convert|replace|make)\w*\b.{0,35}\b(?:plastic|wood|wooden|metal|metallic|glass|marble|stone|leather)\b.{0,45}\b(?:to|into|with|from|back to)\b",
        ),
    ),
    TaskRule(
        "recolor_or_palette",
        "Change local colors or an overall color palette/theme.",
        (
            r"\b(?:recolor|color palette|colour palette|color scheme|colour scheme|color theme|colour theme)\w*\b",
            r"\b(?:change|restore|revert)\b.*\bcolou?r\b",
            r"\bcolou?r\b.*\b(?:change|restore|revert|shift)\b",
            r"\b(?:red|blue|green|yellow|orange|purple|pink|brown|black|white|grey|gray|teal|cyan|magenta|gold|silver)\b.*\b(?:instead of|back to|to a|into a)\b",
        ),
    ),
    TaskRule(
        "object_addition",
        "Add a new object, graphic, decoration, or scene element.",
        (
            r"^(?:please )?(?:add|insert|introduce|include)\b",
            r"\b(?:add|insert|introduce)\b.{0,80}\b(?:new|additional|another|extra|more)\b",
        ),
    ),
    TaskRule(
        "cleanup_damage_or_aging",
        "Clean dirt/damage/aging artifacts or restore a damaged/dirty subject.",
        (
            r"\b(?:clean|remove|erase|repair|restore|fix)\w*\b.{0,90}\b(?:dirt|mud|stain|smudge|scratch|crack|chip|peeling|wrinkle|crease|dust|grime|rust|damage|weathering|wear|tear|blemish|cobweb)s?\b",
            r"\b(?:dirt|mud|stain|smudge|scratch|crack|chip|peeling|wrinkle|crease|dust|grime|rust|damage|weathering|wear|tear|blemish|cobweb)s?\b.{0,90}\b(?:clean|remove|erase|repair|restore|fix)\w*\b",
        ),
    ),
    TaskRule(
        "reflection_shadow_or_glow",
        "Add/remove/change reflections, shadows, glow, sparkle, or related optical effects.",
        (
            r"\b(?:add|remove|change|restore|reduce|increase|soften|darken|brighten|apply|eliminate)\w*\b.{0,40}\b(?:reflection|reflections|reflective|shadow|shadows|drop shadow|glow|glowing|sparkle|sparkles|glitter|neon glow|halo)\b",
            r"\b(?:reflection|reflections|reflective|shadow|shadows|drop shadow|glow|glowing|sparkle|sparkles|glitter|neon glow|halo)\b.{0,55}\b(?:add|remove|change|restore|reduce|increase|soften|darken|brighten|apply|eliminate|from|to)\w*\b",
        ),
    ),
    TaskRule(
        "transparency_or_occlusion",
        "Change transparency, opacity, visibility-through, or foreground occlusion.",
        (
            r"\b(?:transparent|transparency|translucent|opacity|opaque|see[- ]through|occlud|obscur)\w*\b",
        ),
    ),
    TaskRule(
        "background_or_environment",
        "Replace, remove, isolate, or otherwise change the scene background/environment.",
        (
            r"\b(?:replace|change|restore|revert|convert|set|make)\w*\b.{0,60}\b(?:the |entire |whole |scene )?(?:background|backdrop|environment|surroundings|setting)\b(?!\s+(?:shoe|object|person|man|woman|figure|car|building|item|subject))",
            r"\bremove\w*\b.{0,20}\b(?:the |entire |whole |scene )?(?:background|backdrop|environment|surroundings|setting)\b",
            r"\b(?:the |entire |whole |scene )?(?:background|backdrop|environment|surroundings|setting)\b(?!\s+(?:shoe|object|person|man|woman|figure|car|building|item|subject)).{0,90}\b(?:with|to|into|from|replace|change|remove|restore|revert)\b",
            r"\bisolate\b.{0,80}\b(?:on|onto|against)\b.{0,40}\b(?:background|backdrop)\b",
        ),
    ),
    TaskRule(
        "object_replacement",
        "Replace or swap an object/identity with a different object/identity.",
        (
            r"\b(?:replace|swap|substitute)\b",
            r"\bturn\b.*\binto\b",
        ),
    ),
    TaskRule(
        "object_removal",
        "Remove a visible object, person, text/graphic element, or decoration.",
        (
            r"^(?:please )?(?:remove|erase|delete|eliminate|take out)\b",
            r"\b(?:remove|erase|delete|eliminate)\b.*\b(?:object|person|man|woman|logo|icon|decoration|element|item)\b",
        ),
    ),
    TaskRule(
        "shape_geometry_or_deformation",
        "Change shape, proportions, curvature, deformation, or geometric structure.",
        (
            r"\b(?:shape|reshape|geometry|geometric|deform|distort|warp|bend|curv|straighten|flatten|rounder|thicker|thinner|wider|narrower)\w*\b",
        ),
    ),
    TaskRule(
        "visual_effect_or_overlay",
        "Add/remove a non-photographic overlay or visual effect not covered above.",
        (
            r"\b(?:effect|overlay|double exposure|vignette|border|frame|graphic|pattern|particles?|confetti|fire|flames?|smoke|water droplets?|motion lines?)\b",
        ),
    ),
)


COMPILED_RULES: tuple[tuple[TaskRule, tuple[re.Pattern[str], ...]], ...] = tuple(
    (rule, tuple(re.compile(pattern, re.IGNORECASE) for pattern in rule.patterns))
    for rule in TASK_RULES
)


def classify_instruction(instruction: str) -> tuple[str, str | None]:
    normalized = " ".join(instruction.split())
    for rule, patterns in COMPILED_RULES:
        for pattern in patterns:
            if pattern.search(normalized):
                return rule.name, pattern.pattern
    return "other", None


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an S3 URI: {uri}")
    bucket, separator, key = uri[5:].partition("/")
    if not bucket or not separator or not key:
        raise ValueError(f"invalid S3 URI: {uri}")
    return bucket, key.rstrip("/") + "/"


def list_ground_truth_ids(
    root_s3: str, *, profile: str | None, region: str
) -> dict[str, str]:
    import boto3

    bucket, root_key = parse_s3_uri(root_s3)
    prefix = f"{root_key}shard_00/cases/"
    session = boto3.Session(profile_name=profile, region_name=region)
    client = session.client("s3", region_name=region)
    result: dict[str, str] = {}
    pattern = re.compile(
        rf"^{re.escape(prefix)}([^/]+)/target_json\.with_inline_edit\.json$"
    )
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item.get("Key", "")
            match = pattern.fullmatch(key)
            if match is not None:
                result[match.group(1)] = f"s3://{bucket}/{key}"
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            sample_id = row.get("sample_id")
            instruction = row.get("instruction")
            references = row.get("reference_images")
            target_uri = row.get("target_gcs_uri")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{path}:{line_number} has no sample_id")
            if sample_id in seen:
                raise ValueError(f"duplicate sample_id: {sample_id}")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"{path}:{line_number} has no instruction")
            if not isinstance(references, list) or not references:
                raise ValueError(f"{path}:{line_number} has no reference images")
            if not isinstance(target_uri, str) or not target_uri.startswith("gs://"):
                raise ValueError(f"{path}:{line_number} has invalid target_gcs_uri")
            seen.add(sample_id)
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            handle.write(payload)
            digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-s3", default=DEFAULT_SOURCE_S3)
    parser.add_argument("--ground-truth-s3-prefix", default=DEFAULT_SOURCE_S3)
    parser.add_argument("--aws-profile")
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument("--samples-per-task", type=int, default=50)
    parser.add_argument("--min-tasks", type=int, default=20)
    parser.add_argument("--max-tasks", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples_per_task <= 0:
        raise ValueError("--samples-per-task must be positive")
    if not 1 <= args.min_tasks <= args.max_tasks:
        raise ValueError("expected 1 <= --min-tasks <= --max-tasks")

    input_path = args.input_jsonl.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(input_path)
    ground_truth = list_ground_truth_ids(
        args.ground_truth_s3_prefix,
        profile=args.aws_profile,
        region=args.aws_region,
    )

    all_counts: Counter[str] = Counter()
    available_counts: Counter[str] = Counter()
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    classified_rows: list[dict[str, Any]] = []
    for row in rows:
        task, matched_rule = classify_instruction(row["instruction"])
        all_counts[task] += 1
        sample_id = row["sample_id"]
        gt_uri = ground_truth.get(sample_id)
        classified = {
            **row,
            "eval_task": task,
            "eval_task_version": TAXONOMY_VERSION,
            "eval_task_matched_rule": matched_rule,
            "ground_truth_s3_uri": gt_uri,
        }
        classified_rows.append(classified)
        if gt_uri is not None:
            available_counts[task] += 1
            pools[task].append(classified)

    eligible = [
        rule.name
        for rule in TASK_RULES
        if available_counts[rule.name] >= args.samples_per_task
    ]
    if not args.min_tasks <= len(eligible) <= args.max_tasks:
        counts = {name: available_counts[name] for name in eligible}
        raise ValueError(
            f"eligible task count {len(eligible)} is outside "
            f"[{args.min_tasks}, {args.max_tasks}]: {counts}"
        )

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for task in eligible:
        pool = sorted(pools[task], key=lambda item: item["sample_id"])
        task_rows = rng.sample(pool, args.samples_per_task)
        task_rows.sort(key=lambda item: item["sample_id"])
        for selection_index, row in enumerate(task_rows):
            if row["sample_id"] in selected_ids:
                raise AssertionError(f"duplicate selected sample: {row['sample_id']}")
            selected_ids.add(row["sample_id"])
            selected.append(
                {
                    **row,
                    "eval_selection_index_within_task": selection_index,
                    "eval_seed": args.seed,
                    "source_dataset_s3": args.source_s3,
                }
            )

    expected_total = len(eligible) * args.samples_per_task
    if len(selected) != expected_total:
        raise AssertionError(f"selected {len(selected)} != expected {expected_total}")
    if not 1000 <= expected_total <= 1500:
        raise ValueError(f"evaluation set size {expected_total} is outside [1000, 1500]")

    selected_path = output_dir / "evaluation_manifest.jsonl"
    classified_path = output_dir / "classification_audit.jsonl"
    selected_sha256 = write_jsonl(selected_path, selected)
    classified_sha256 = write_jsonl(classified_path, classified_rows)
    descriptions = {rule.name: rule.description for rule in TASK_RULES}
    summary = {
        "format_version": "instruction_stratified_eval_v1",
        "taxonomy_version": TAXONOMY_VERSION,
        "source_manifest": str(input_path),
        "source_dataset_s3": args.source_s3,
        "ground_truth_s3_prefix": args.ground_truth_s3_prefix,
        "source_rows": len(rows),
        "ground_truth_available_rows": len(ground_truth),
        "ground_truth_missing_rows": len(rows) - len(ground_truth),
        "seed": args.seed,
        "samples_per_task": args.samples_per_task,
        "classification_contract": {
            "method": "ordered_regex_first_match",
            "mutually_exclusive_primary_task": True,
            "multi_edit_policy": (
                "Each instruction receives the first matching primary task, so a case "
                "can be selected at most once even when it requests multiple edits."
            ),
            "label_origin": "heuristic_instruction_taxonomy_not_source_metadata",
            "eligibility_policy": (
                "Only tasks with at least samples_per_task cases that have ground truth "
                "are eligible for stratified sampling."
            ),
        },
        "eligible_task_count": len(eligible),
        "selected_rows": len(selected),
        "eligible_tasks": [
            {
                "task": task,
                "description": descriptions[task],
                "source_count": all_counts[task],
                "ground_truth_available_count": available_counts[task],
                "selected_count": args.samples_per_task,
            }
            for task in eligible
        ],
        "all_task_counts": dict(sorted(all_counts.items())),
        "ground_truth_available_task_counts": dict(sorted(available_counts.items())),
        "outputs": {
            "evaluation_manifest": str(selected_path),
            "evaluation_manifest_sha256": selected_sha256,
            "classification_audit": str(classified_path),
            "classification_audit_sha256": classified_sha256,
        },
        "validation": {
            "unique_source_sample_ids": len({row["sample_id"] for row in rows})
            == len(rows),
            "unique_selected_sample_ids": len(selected_ids) == len(selected),
            "every_selected_row_has_ground_truth": all(
                row["ground_truth_s3_uri"] for row in selected
            ),
            "every_eligible_task_has_exact_sample_count": all(
                sum(row["eval_task"] == task for row in selected)
                == args.samples_per_task
                for task in eligible
            ),
            "selected_size_in_requested_range": 1000 <= len(selected) <= 1500,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
