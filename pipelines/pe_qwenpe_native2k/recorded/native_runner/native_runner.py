"""PCB adapter using the frozen training input and sampling implementations."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import runtime_helpers as H

def read(p): return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def th(x):return hashlib.sha256(x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def rows(a):
    result=[json.loads(x) for x in (a.root/'inputs/manifest.jsonl').read_text().splitlines()]
    if a.ids:result=[r for r in result if r['request_id'] in a.ids.split(',')]
    else:result=[r for i,r in enumerate(result) if i%a.shards==a.shard]
    return result
def config(a):return read(a.root/'expected-resume-contract.json')['effective_config']
def encoding_identity(a):
    c=config(a); value={'model_inventory':sha(a.root/'model-inventory.json'),'manifest':sha(a.root/'inputs/manifest.jsonl'),'conditioner':c['model']['conditioner'],'vae':c['model']['vae'],
       'fp8':c['model']['te_fp8_config'],'data':{k:c['data']['train_dataset']['init_args'][k] for k in ('variant','image_max_pixels','reference_region_allocation_max_pixels','pack_target_T','pixel_patch_size')},
       'commit':'b1d83ee3db96b7ee2d0ece062413dd07dfde82c1','runner':sha(Path(__file__)),'helper':sha(Path(H.__file__)),'recipe':'Float8CurrentScaling'}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def prepare(context,row,root):
    from src.data.dataset.craft_v1_edit_mds_annotation import adapt_native_edit_annotation
    from src.data.dataset.craft_v1_image_edit_packing import build_craft_v1_edit_inference_batch
    from src.inference.craft_v1_edit import _segment_layout
    source=root/'inputs'/row['native_source_path']; annotation=root/'inputs'/row['annotation_path']
    assert sha(source)==row['native_source_sha256'] and sha(annotation)==row['annotation_sha256']
    from PIL import Image
    with Image.open(source) as im:
        assert list(im.size)==row['native_source_size'] and im.width%32==im.height%32==0 and im.width*im.height<=4194304
    c=context.config; d=c['data']['train_dataset']['init_args']; n=c['model']['denoiser']['init_args']
    batch=build_craft_v1_edit_inference_batch(conditioner=context.conditioner,annotation=adapt_native_edit_annotation(read(annotation)),
       reference_images=[source.read_bytes()],output_width=row['expected_output_width'],output_height=row['expected_output_height'],
       variant=d['variant'],pack_target_T=n['pack_target_T'],pixel_patch_size=d['pixel_patch_size'],image_max_pixels=d['image_max_pixels'],
       reference_region_allocation_max_pixels=d['reference_region_allocation_max_pixels'],max_reference_images=d['max_reference_images'],
       reference_frame_codes=tuple(n['element_bbox_rope_reference_frame_codes']))
    target,lens=_segment_layout(batch); m=batch.metadata[0];pos=int(batch.text_lens[0]);weak=int(m['cfg_null_qwen_text_len'])
    save(root/'inputs/cases'/row['request_id']/'model-prompts.json',{'positive':batch.captions[0],'weak':m['cfg_null_caption']})
    assert batch.n_samples==1 and not torch.count_nonzero(batch.img_patches[target])
    assert 'post_encoder_text_keep_mask' not in m
    assert min(pos,weak)>0 and max(pos,weak)<11776, f'Text admission failed: positive={pos}, weak={weak}'
    assert lens[0]+max(pos,weak)<=22528, 'Pack admission exceeded'
    audit={'request_id':row['request_id'],'positive_text_tokens':pos,'weak_text_tokens':weak,'strict_pack_cost':lens[0]+max(pos,weak),
       'output_width':row['expected_output_width'],'output_height':row['expected_output_height'],'seed':row['generation_seed'],
       'source_sha256':row['source_sha256'],'native_source_sha256':row['native_source_sha256'],'annotation_sha256':row['annotation_sha256'],'variant':d['variant'],'text_cap':11776,
       'pack_target_T':22528,'target_pixels_used':False,'truncated':False,'image_grids':batch.img_grids.tolist(),
       'roi_tokens':[int(x) for x in m.get('reference_region_token_counts',[])],
       'positive_prompt_sha256':hashlib.sha256(batch.captions[0].encode()).hexdigest(),
       'weak_prompt_sha256':hashlib.sha256(m['cfg_null_caption'].encode()).hexdigest()}
    for k in ('qwen_input_ids','cfg_null_qwen_input_ids','text_pos_meta','cfg_null_text_pos_meta'):
        if isinstance(m.get(k),torch.Tensor):audit[k+'_sha256']=th(m[k])
    return batch,audit

def audit(a):
    from transformers import AutoTokenizer
    torch.set_num_threads(2);c=config(a);settings=copy.deepcopy(c['model']['conditioner']['init_args']);settings.pop('weight_path')
    settings['reference_region_visual_stride']=32
    ctx=H.EncodingContext(c,torch.device('cpu'),conditioner=SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(str(a.models/'Qwen3.8-27B'),local_files_only=True),**settings))
    accepted=[];rejected=[]
    for row in rows(a):
        try:
            batch,e=prepare(ctx,row,a.root);accepted.append(e)
            save(a.root/'audits'/f"{row['request_id']}.json",e)
            del batch
        except (ValueError,AssertionError) as error:
            rejected.append({'request_id':row['request_id'],'error':str(error)})
        if (len(accepted)+len(rejected))%100==0:print('AUDITED',len(accepted),'rejected',len(rejected),flush=True)
    assert len(accepted)+len(rejected)==len(rows(a))
    canary=set()
    for key in ('positive_text_tokens','strict_pack_cost'):
        if accepted:canary.add(max(accepted,key=lambda r:r[key])['request_id'])
    if accepted:
        canary.add(max(accepted,key=lambda r:len(r['roi_tokens']))['request_id'])
        canary.add(max(accepted,key=lambda r:max(r['output_width']/r['output_height'],r['output_height']/r['output_width']))['request_id'])
        canary.add(accepted[0]['request_id'])
    if any(r['request_id']=='g2__chart_editing_000020' for r in accepted):canary.add('g2__chart_editing_000020')
    result={'status':'PASS' if not rejected else 'ADMISSION_REJECTIONS','encoding_identity':encoding_identity(a),'accepted':len(accepted),'rejected':rejected,'canary_ids':sorted(canary),'max_positive':max((r['positive_text_tokens'] for r in accepted),default=0),'max_pack':max((r['strict_pack_cost'] for r in accepted),default=0)}
    save(a.root/(f'audit-shard-{a.shard}.json' if a.shards>1 else 'audit.json'),result);print(json.dumps(result),flush=True)

def native_bundle(context,batch,row):
    """Use native _compute_cond_bundle and capture the native sampler arguments."""
    from src.native_task import PixelCraftNativeTask as T
    from src.diffusion.flow_matching.training_self_flow_varlen import SelfFlowVarlenTrainer as D
    from src.inference.craft_v1_edit import sample_training_aligned_edit_batch
    class Task:
        _stack_packed_metadata=staticmethod(T._stack_packed_metadata)
        _apply_post_encoder_text_drop=staticmethod(T._apply_post_encoder_text_drop)
        _flatten_text_tokens=staticmethod(T._flatten_text_tokens)
        _ragged_input_ids_from_pack=staticmethod(T._ragged_input_ids_from_pack)
        _encode_varlen_images=T._encode_varlen_images
        _compute_cond_bundle=T._compute_cond_bundle
    class Capture:
        num_steps=50
        def sample_varlen_edit(self,net,**kwargs):self.kwargs=kwargs;return [torch.zeros(1)]
    task=Task();task.conditioner=context.conditioner;task.vae=context.vae;task._te_cond_fp8_applied=True
    task.device=context.device;task.denoiser=object();task.ema_denoiser=object();task.diffusion_sampler=Capture()
    task.diffusion_trainer=SimpleNamespace(cfg_dropout_mode='multimodal_text_null',_gather_text_pos_meta_valid=D._gather_text_pos_meta_valid,_gather_text_corr_mem_k_valid=D._gather_text_corr_mem_k_valid)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        sample_training_aligned_edit_batch(task,batch,guidance=2.5,num_steps=50,noise_scale=1.0,seed=row['generation_seed'],use_ema=True,decode=False,use_reference_kv_cache=True)
    bundle=task.diffusion_sampler.kwargs
    for k,v in bundle.items():
        if isinstance(v,torch.Tensor) and v.is_floating_point():assert torch.isfinite(v).all(),k
    return bundle

def bundle_path(a,row):return a.root/'bundles'/f"{row['request_id']}.pt"
def verify_bundle(a,row):
    p=bundle_path(a,row);m=read(p.with_suffix('.json'))
    assert m['encoding_identity']==encoding_identity(a) and m['sha256']==sha(p)
    assert m['request_id']==row['request_id'];return p

def encode(a):
    if not rows(a):return
    torch.set_num_threads(4);torch.cuda.set_device(0)
    assert torch.cuda.get_device_capability(0)[0]==9,'H100 runtime required'
    ctx=H.load_conditioner(config(a),a.models,'cuda:0',text_cap=11776)
    assert ctx.evidence['fp8_recipe_class']=='Float8CurrentScaling'
    save(a.root/'evidence'/f'encoder-{a.shard}.json',ctx.evidence)
    rejected={r['request_id'] for r in read(a.root/'audit.json')['rejected']}
    for row in rows(a):
        if row['request_id'] in rejected:continue
        path=bundle_path(a,row)
        if path.with_suffix('.json').exists():verify_bundle(a,row);continue
        batch,case=prepare(ctx,row,a.root);assert case==read(a.root/'audits'/f"{row['request_id']}.json")
        bundle=native_bundle(ctx,batch,row)
        cpu={k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in bundle.items()}
        path.parent.mkdir(exist_ok=True);tmp=path.with_suffix('.partial');torch.save({'bundle':cpu,'case':case},tmp);tmp.replace(path)
        restored=torch.load(path,map_location='cpu',weights_only=False)
        for k,v in cpu.items():
            if isinstance(v,torch.Tensor):assert torch.equal(v,restored['bundle'][k]),k
            else:assert v==restored['bundle'][k],k
        save(path.with_suffix('.json'),{'request_id':row['request_id'],'encoding_identity':encoding_identity(a),'sha256':sha(path),'native_sampler_kwargs_captured':True,'serialization_roundtrip_exact':True,'case':case})
        print('ENCODED',row['request_id'],flush=True)
        del restored,bundle,cpu,batch;gc.collect()

def restore(a,model):
    from torch.distributed.checkpoint import FileSystemReader,load_state_dict
    ckpt=a.root/'checkpoints'/f'step{a.step}'
    plan=read(ckpt/'CHECKPOINT.json');assert plan['step']==a.step and plan['weights']=='EMA'
    reader=FileSystemReader(ckpt);meta=reader.read_metadata();prefix='state_dict.ema_denoiser.'
    entries={k[len(prefix):]:v for k,v in meta.state_dict_metadata.items() if k.startswith(prefix)}
    state=model.state_dict();assert len(entries)==len(state)==463;assert set(entries)==set(state)
    assert all(tuple(entries[k].size)==tuple(v.shape) for k,v in state.items())
    selected={'ema_denoiser.'+k:v for k,v in state.items()}
    load_state_dict({'state_dict':selected},storage_reader=reader,no_dist=True)
    model.load_state_dict({k.removeprefix('ema_denoiser.'):v for k,v in selected.items()},strict=True)
    return {'step':a.step,'weights':'EMA','keys':len(state),'strict':True,'checkpoint_sha256':sha(ckpt/'CHECKPOINT.json'),'optimizer_loaded':False}

def infer(a):
    from src.diffusion.flow_matching.sampling import EulerSamplerJiT
    from src.diffusion.flow_matching.scheduling import LinearScheduler
    if not rows(a):return
    torch.set_num_threads(4);torch.cuda.set_device(0);c=config(a);spec=c['model']['denoiser']
    with torch.device('cpu'),H._default_dtype(torch.bfloat16):model=H._class(spec['class_path'])(**copy.deepcopy(spec['init_args']))
    model.requires_grad_(False).eval();proof=restore(a,model)
    assert model.conditioning_policy=='reference_only' and model.pack_target_T==22528 and model.strict_pack_target
    model.to('cuda');ctx=H.load_vae_only(c,a.models,'cuda:0');out=a.root/'runs'/f'step{a.step}'
    save(out/'evidence'/f'restore-{a.shard}.json',proof)
    rejected={r['request_id'] for r in read(a.root/'audit.json')['rejected']}
    host={'img_grids','img_seg_sample_id','img_seg_frame','img_seg_is_target'}
    for row in rows(a):
        if row['request_id'] in rejected:continue
        dest=out/'images'/f"{row['request_id']}.png";receipt=dest.with_suffix('.json')
        if receipt.exists():
            r=read(receipt);assert r['step']==a.step and r['image_sha256']==sha(dest) and r['encoding_identity']==encoding_identity(a);continue
        p=verify_bundle(a,row);v=torch.load(p,map_location='cpu',weights_only=False)
        bundle={k:x.to('cuda') if isinstance(x,torch.Tensor) and k not in host else x for k,x in v['bundle'].items()}
        assert bundle['guidance']==2.5 and bundle['noise_scale']==1.0 and bundle['seed']==row['generation_seed']
        sampler=EulerSamplerJiT(num_steps=50,timeshift=4.6,prediction_type='v',guidance=2.5,scheduler=LinearScheduler())
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            latents=sampler.sample_varlen_edit(model,**bundle)
            assert all(torch.isfinite(x).all() for x in latents)
            im=H.decode(ctx,latents,v['case'])
        dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.partial');im.save(tmp,format='PNG');tmp.replace(dest)
        save(receipt,{'request_id':row['request_id'],'step':a.step,'weights':'EMA','cfg':2.5,'timeshift':4.6,'input_order':'trainorder-v2','steps':50,'seed':row['generation_seed'],'image_sha256':sha(dest),'encoding_identity':encoding_identity(a),'bundle_sha256':sha(p),'checkpoint_sha256':proof['checkpoint_sha256'],'finite_latents':True,'case':v['case']})
        print('GENERATED',a.step,row['request_id'],flush=True);del bundle,latents,v;gc.collect()

def verify(a):
    from PIL import Image
    rejected={r['request_id'] for r in read(a.root/'audit.json')['rejected']};records=[]
    for row in rows(a):
        if row['request_id'] in rejected:continue
        p=a.root/'runs'/f'step{a.step}'/'images'/f"{row['request_id']}.png";r=read(p.with_suffix('.json'))
        assert r['image_sha256']==sha(p) and r['step']==a.step and r['weights']=='EMA' and r['encoding_identity']==encoding_identity(a)
        assert r['cfg']==2.5 and r['steps']==50 and r['seed']==row['generation_seed'] and r['finite_latents']
        with Image.open(p) as im:im.load();assert im.size==(row['expected_output_width'],row['expected_output_height'])
        records.append(r)
    result={'status':'PASS','step':a.step,'images':len(records),'rejected':sorted(rejected),'encoding_identity':encoding_identity(a),'publish_to_leaderboard':False}
    save(a.root/'runs'/f'step{a.step}'/('CANARY.json' if a.ids else 'COMPLETE.json'),result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit','encode','infer','verify']);p.add_argument('--root',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--step',type=int);p.add_argument('--ids');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();globals()[a.mode](a)
