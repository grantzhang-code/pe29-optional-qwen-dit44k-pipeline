"""Copy only original DCP EMA storage ranges, preserving serialized tensor bytes."""
from common import ROOT,CFG,client,save,sha,publish
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit
from collections import defaultdict
from dataclasses import replace
import copy,hashlib,pickle,torch

s=client();u=urlsplit(CFG['dit_checkpoint_uri']);base=u.path.lstrip('/')
dest=ROOT/'checkpoint-ema';dest.mkdir(exist_ok=True)
meta=pickle.loads((ROOT/'.metadata').read_bytes())
selected={k:v for k,v in meta.state_dict_metadata.items() if k.startswith(CFG['dit_weight_namespace'])}
assert len(selected)==463
assert all(v.properties.dtype==torch.float32 for v in selected.values())
groups=defaultdict(list)
for k,v in meta.storage_data.items():
    if k.fqn in selected:groups[v.relative_path].append((k,v))
def one(item):
    name,entries=item;head=s.head_object(Bucket=u.netloc,Key=base+name)
    assert head['ETag']==CFG['source_objects'][name]['etag'] and head['ContentLength']==CFG['source_objects'][name]['size']
    path=dest/name;tmp=path.with_suffix('.partial');mappings={};chunks=[]
    with tmp.open('wb') as f:
        for index,entry in sorted(entries,key=lambda item:item[1].offset):
            blob=s.get_object(Bucket=u.netloc,Key=base+name,IfMatch=head['ETag'],Range=f'bytes={entry.offset}-{entry.offset+entry.length-1}')['Body'].read()
            assert len(blob)==entry.length
            offset=f.tell();f.write(blob)
            mappings[index]=replace(entry,offset=offset)
            chunks.append({'fqn':index.fqn,'tensor_offsets':list(index.offset),'source_offset':entry.offset,'compact_offset':offset,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()})
    tmp.replace(path)
    # Upload each complete object; readers wait for the final committed proof.
    publish(path,'models/dit-ema/'+name)
    return mappings,dict(name=name,bytes=path.stat().st_size,sha256=sha(path),source_etag=head['ETag'],chunks=chunks)
maps={};files=[]
with ThreadPoolExecutor(max_workers=16) as pool:
    for mapping,file in pool.map(one,groups.items()):
        maps.update(mapping);files.append(file);print('EMA_FILE_READY',len(files),len(groups),flush=True)
compact=copy.copy(meta);compact.state_dict_metadata=selected;compact.planner_data={k:v for k,v in meta.planner_data.items() if k in selected};compact.storage_data=maps
(dest/'.metadata').write_bytes(pickle.dumps(compact));(dest/'meta.pt').write_bytes((ROOT/'meta.pt').read_bytes())
control=torch.load(dest/'meta.pt',map_location='cpu',weights_only=False);assert control['global_step']==6000
save(ROOT/'expected-resume-contract.json',control['mariana_resume_contract'])
for name in ['.metadata','meta.pt']:
    p=dest/name;publish(p,'models/dit-ema/'+name);files.append(dict(name=name,bytes=p.stat().st_size,sha256=sha(p)))
proof=dict(status='PASS',step=6000,weights='EMA',source_uri=CFG['dit_checkpoint_uri'],namespace=CFG['dit_weight_namespace'],keys=463,source_metadata_sha256=sha(ROOT/'.metadata'),source_meta_sha256=sha(ROOT/'meta.pt'),files=files,serialized_tensor_bytes_unchanged=True,optimizer_loaded=False)
save(dest/'CHECKPOINT.json',proof);publish(dest/'CHECKPOINT.json','models/dit-ema/CHECKPOINT.json');publish(ROOT/'expected-resume-contract.json','models/expected-resume-contract.json')
save(ROOT/'DIT_READY.json',dict(status='PASS',keys=463,checkpoint_sha256=sha(dest/'CHECKPOINT.json'),files=len(files),bytes=sum(f['bytes'] for f in files)))
publish(ROOT/'DIT_READY.json','models/DIT_READY.json')
print('DIT_READY',flush=True)
