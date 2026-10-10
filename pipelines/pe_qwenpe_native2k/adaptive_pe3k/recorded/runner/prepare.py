"""Freeze ordered PE JSON, source-only pixels and native 2K geometry."""
from pathlib import Path
from urllib.parse import urlsplit
import argparse, hashlib, json, math, shutil
from PIL import Image
from adapt_pe_to_native import adapt
from canvas_policy import editor_canvas_size

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def dimensions(w,h,upscale):
    ratio=math.sqrt(4194304/(w*h))
    if not upscale:ratio=min(1,ratio)
    width,height=max(32,round(w*ratio)//32*32),max(32,round(h*ratio)//32*32)
    while width*height>4194304:
        if width>=height:width-=32
        else:height-=32
    return width,height

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode');ap.add_argument('--root',type=Path,required=True);a,_=ap.parse_known_args()
    root=a.root;cache=Path(__file__).resolve().parents[1]/'package/workflow/source-cache'
    predictions=[json.loads(l) for l in (root/'caption/predictions.jsonl').read_text().splitlines()]
    manifest=[];excluded=[]
    for p in predictions:
        rid=p['selection']['case_id']
        assert rid and '/' not in rid and '..' not in rid
        try:
            if p.get("repetition_rejected"):
                raise ValueError("repetition loop remains after official-sampling retry")
            if p.get('json_error') or p.get('schema_error') or p.get('finish_reason')=='length':
                raise ValueError(f"caption rejected: {p.get('json_error')} {p.get('schema_error')} finish={p.get('finish_reason')}")
            annotation=adapt(p['prediction'])
        except (ValueError,TypeError,KeyError) as e:
            excluded.append({'request_id':rid,'input_index':p['input_index'],'sample_id':p['sample_id'],'error':str(e)});continue
        d=root/'inputs/cases'/rid;d.mkdir(parents=True,exist_ok=True)
        source=cache/Path(urlsplit(p['source_s3_uri']).path).name
        assert sha(source)==p['selection']['source_sha256']
        out=d/'source.png'
        if not out.exists():out.symlink_to(source)
        with Image.open(source) as im:
            im=im.convert('RGB');original_size=list(im.size);native_size=dimensions(*im.size,False)
            if im.size!=native_size:im=im.resize(native_size,Image.Resampling.LANCZOS)
            im.save(d/'source-native.png')
        canvas=p['target_canvas_size'];w,h=(2*d for d in editor_canvas_size(p))
        write(d/'annotation.json',annotation);write(d/'pe.json',p['prediction'])
        manifest.append(dict(request_id=rid,sample_id=p['sample_id'],input_index=p['input_index'],task=p['selection']['group'],origin=p['selection']['origin'],
            original_instruction=p['original_instruction'],instruction=p['instruction'],source_s3_uri=p['source_s3_uri'],
            source_path=f'cases/{rid}/source.png',source_sha256=sha(out),native_source_path=f'cases/{rid}/source-native.png',native_source_sha256=sha(d/'source-native.png'),
            original_source_size=original_size,native_source_size=list(native_size),native_resize_policy='original_bytes_area_le_4194304_downscale_only_floor32_lanczos_png_v1',
            annotation_path=f'cases/{rid}/annotation.json',annotation_sha256=sha(d/'annotation.json'),pe_path=f'cases/{rid}/pe.json',pe_sha256=sha(d/'pe.json'),
            expected_output_width=w,expected_output_height=h,generation_seed=42,caption_preparation_version='trainorder-v2-native2k-v1',
            target_canvas_size=canvas,use_qwen_pe=True,instruction_policy=p["instruction_policy"],qwen_pe_predictions_sha256=p["selection"]["qwen_pe_predictions_sha256"],rewritten_instruction_sha256=p["selection"]["rewritten_instruction_sha256"]))
    assert len(manifest)+len(excluded)==len(predictions)
    for p,values in [(root/'inputs/manifest.jsonl',manifest),(root/'caption/excluded.jsonl',excluded)]:
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values))
    print(json.dumps({'prepared':len(manifest),'excluded':len(excluded)}),flush=True)

if __name__=='__main__':main()
