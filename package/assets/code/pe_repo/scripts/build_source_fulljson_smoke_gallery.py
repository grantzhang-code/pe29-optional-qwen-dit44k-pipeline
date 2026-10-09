#!/usr/bin/env python3
"""Build a self-contained source-only gallery for full-JSON smoke results."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlparse

from PIL import Image, ImageOps


THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
FENCE_START_RE = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)
FENCE_END_RE = re.compile(r"\s*```\s*$")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"invalid S3 URI: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def read_uri(uri: str, client: Any) -> bytes:
    if uri.startswith("s3://"):
        bucket, key = parse_s3_uri(uri)
        return client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return Path(uri).read_bytes()


def read_jsonl(uri: str, client: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(read_uri(uri, client).decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{uri}:{line_number}: row must be an object")
        rows.append(value)
    return rows


def parse_response(response: Any) -> dict[str, Any]:
    text = THINK_RE.sub("", str(response or ""), count=1)
    text = FENCE_START_RE.sub("", text, count=1)
    text = FENCE_END_RE.sub("", text, count=1).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


def response_elements(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    composition = value.get("compositional_deconstruction")
    if not isinstance(composition, Mapping):
        return []
    elements = composition.get("elements")
    if not isinstance(elements, list):
        return []
    return [dict(item) for item in elements if isinstance(item, Mapping)]


def valid_bbox(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
        and 0 <= value[0] < value[2] <= 1000
        and 0 <= value[1] < value[3] <= 1000
    )


def thumbnail_data_uri(payload: bytes, *, max_dimension: int) -> tuple[str, int, int]:
    with Image.open(io.BytesIO(payload)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="WEBP", quality=82, method=4)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/webp;base64,{encoded}", width, height


def reject_target_fields(value: Any, *, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered.startswith("target") or lowered in {"edited_image", "output_image"}:
                raise ValueError(f"source-only guard rejected {path}.{key}")
            reject_target_fields(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_target_fields(child, path=f"{path}[{index}]")


def build(args: argparse.Namespace) -> dict[str, Any]:
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        region_name=args.aws_region,
        config=Config(max_pool_connections=args.workers, retries={"max_attempts": 8, "mode": "adaptive"}),
    )
    manifest = read_jsonl(args.manifest, client)
    results = read_jsonl(args.results, client)
    if len(manifest) != args.expected_count or len(results) != args.expected_count:
        raise ValueError(
            f"row count mismatch: manifest={len(manifest)} results={len(results)} "
            f"expected={args.expected_count}"
        )
    reject_target_fields(manifest)
    by_index = {int(row["input_index"]): row for row in results}
    if set(by_index) != set(range(args.expected_count)):
        raise ValueError("results do not contain each input_index exactly once")

    def load_image(item: tuple[int, Mapping[str, Any]]) -> tuple[int, str, int, int]:
        index, row = item
        uri = str(row["s3_image_path"])
        data_uri, width, height = thumbnail_data_uri(
            read_uri(uri, client), max_dimension=args.max_dimension
        )
        return index, data_uri, width, height

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        images = {
            index: (data_uri, width, height)
            for index, data_uri, width, height in executor.map(load_image, enumerate(manifest))
        }

    cards: list[dict[str, Any]] = []
    valid_json_count = 0
    valid_bbox_count = 0
    invalid_bbox_count = 0
    source_counts: dict[str, int] = {}
    for index, manifest_row in enumerate(manifest):
        result = by_index[index]
        source = manifest_row.get("source")
        source = dict(source) if isinstance(source, Mapping) else {}
        dataset_source = str(source.get("dataset_source") or "unknown")
        source_counts[dataset_source] = source_counts.get(dataset_source, 0) + 1
        parsed: dict[str, Any] | None = None
        parse_error: str | None = None
        try:
            parsed = parse_response(result.get("response"))
            valid_json_count += 1
        except Exception as exc:  # noqa: BLE001 - gallery must expose invalid responses
            parse_error = f"{type(exc).__name__}: {exc}"
        elements = response_elements(parsed or {})
        boxes: list[dict[str, Any]] = []
        for element_index, element in enumerate(elements):
            bbox = element.get("bbox")
            if valid_bbox(bbox):
                valid_bbox_count += 1
                boxes.append(
                    {
                        "bbox": bbox,
                        "label": str(element.get("label") or element.get("type") or element_index),
                        "type": str(element.get("type") or ""),
                    }
                )
            elif bbox is not None:
                invalid_bbox_count += 1
        data_uri, width, height = images[index]
        cards.append(
            {
                "index": index,
                "sample_id": source.get("sample_id"),
                "dataset_source": dataset_source,
                "image": data_uri,
                "width": width,
                "height": height,
                "json_valid": parsed is not None,
                "parse_error": parse_error,
                "boxes": boxes,
                "element_count": len(elements),
                "high_level_description": (parsed or {}).get("high_level_description"),
                "style_description": (parsed or {}).get("style_description"),
                "response": parsed if parsed is not None else result.get("response"),
            }
        )

    summary = {
        "rows": len(cards),
        "json_valid": valid_json_count,
        "json_invalid": len(cards) - valid_json_count,
        "valid_bboxes": valid_bbox_count,
        "invalid_bboxes": invalid_bbox_count,
        "source_counts": dict(sorted(source_counts.items())),
        "target_images_displayed": 0,
    }
    payload = json.dumps(
        {"summary": summary, "cards": cards},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__GALLERY_PAYLOAD__", payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    return summary


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IEV6 source-only full-JSON smoke100</title>
<style>
:root{color-scheme:dark;--bg:#101115;--panel:#191b22;--ink:#f4f6fb;--muted:#9aa2b1;--line:#303440;--accent:#65d1ff;--bad:#ff6b7a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 ui-sans-serif,system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;padding:14px 20px;background:#101115ee;border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}
h1{font-size:18px;margin:0 0 8px}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.stats{color:var(--muted)}
input,select,button{background:#22252e;color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:7px 9px}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(390px,1fr));gap:14px;padding:14px}.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.meta{padding:10px 12px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}.tag{border:1px solid var(--line);border-radius:999px;padding:2px 7px;color:var(--muted)}.ok{color:#76e6a5}.bad{color:var(--bad)}
.frame{position:relative;background:#090a0c}.frame img{width:100%;height:auto;display:block}.box{position:absolute;border:1.5px solid var(--accent);pointer-events:none}.box span{position:absolute;left:-1px;top:-18px;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;background:#0587bde8;color:white;font-size:10px;padding:1px 4px}
.desc{padding:10px 12px;color:#cdd2dc;max-height:110px;overflow:auto}.desc strong{color:white}.json{margin:0;border-top:1px solid var(--line);padding:10px 12px;max-height:310px;overflow:auto;white-space:pre-wrap;color:#b9c0ce;font:11px/1.35 ui-monospace,monospace}
.hidden{display:none!important}
</style></head><body><header><h1>IEV6 source-only · C4B-long-MTP3 c10000 · smoke100</h1>
<div class="controls"><input id="q" placeholder="sample id / description"><select id="src"><option value="">all sources</option></select><label><input id="boxes" type="checkbox" checked> boxes</label><label><input id="invalid" type="checkbox"> invalid JSON only</label><span class="stats" id="stats"></span></div></header><main id="grid"></main>
<script>const DATA=__GALLERY_PAYLOAD__;const grid=document.querySelector('#grid'),q=document.querySelector('#q'),src=document.querySelector('#src'),boxes=document.querySelector('#boxes'),invalid=document.querySelector('#invalid'),stats=document.querySelector('#stats');
Object.keys(DATA.summary.source_counts).forEach(x=>src.add(new Option(`${x} (${DATA.summary.source_counts[x]})`,x)));
function esc(x){return String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function render(){let needle=q.value.trim().toLowerCase(),shown=0;grid.innerHTML='';for(const c of DATA.cards){let hay=JSON.stringify([c.sample_id,c.high_level_description,c.response]).toLowerCase();if(needle&&!hay.includes(needle)||src.value&&c.dataset_source!==src.value||invalid.checked&&c.json_valid)continue;shown++;let el=document.createElement('article');el.className='card';let boxHtml=c.boxes.map((b,i)=>{let[x1,y1,x2,y2]=b.bbox;return `<div class="box" style="left:${x1/10}%;top:${y1/10}%;width:${(x2-x1)/10}%;height:${(y2-y1)/10}%;${boxes.checked?'':'display:none'}"><span>${esc(i+' '+b.label)}</span></div>`}).join('');el.innerHTML=`<div class="meta"><b>#${c.index}</b><span class="tag">${esc(c.dataset_source)}</span><span class="tag">${c.width}×${c.height}</span><span class="tag">${c.element_count} elements</span><span class="${c.json_valid?'ok':'bad'}">${c.json_valid?'valid JSON':'invalid JSON'}</span><span class="tag">${esc(c.sample_id)}</span></div><div class="frame"><img loading="lazy" src="${c.image}">${boxHtml}</div><div class="desc"><strong>HLD</strong> ${esc(c.high_level_description||c.parse_error||'—')}</div><pre class="json">${esc(JSON.stringify(c.response,null,2))}</pre>`;grid.appendChild(el)}stats.textContent=`showing ${shown}/${DATA.summary.rows} · valid ${DATA.summary.json_valid} · bbox ${DATA.summary.valid_bboxes} · target images 0`}
[q,src,boxes,invalid].forEach(x=>x.addEventListener('input',render));render();</script></body></html>'''


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--max-dimension", type=int, default=900)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--aws-region", default="us-east-1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
