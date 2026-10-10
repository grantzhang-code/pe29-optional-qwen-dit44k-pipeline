"""Freeze verified PE3k greedy-loop-retry DiT6k outputs against unchanged Qwen controls."""
from pathlib import Path 
import hashlib ,json ,tarfile 
ROOT =Path (__file__ ).resolve ().parent 
SHA =lambda b :hashlib .sha256 (b ).hexdigest ()
ENC =lambda v :(json .dumps (v ,ensure_ascii =False ,sort_keys =True )+'\n').encode ()
read =lambda p :json .loads (p .read_bytes ())
rows =lambda p :[json .loads (l )for l in p .read_bytes ().splitlines ()if l ]
CONTROL ='qwen_image21_cfg1'
def main ():
    assert not (ROOT /'protocol.json').exists ()
    ref =read (ROOT /'inputs/reference-protocol.json');cp =read (ROOT /'inputs/control-protocol.json')
    assert SHA ((ROOT /'inputs/control-pairs.jsonl').read_bytes ())==cp ['input_hashes']['pairs.jsonl']
    assert SHA ((ROOT /'inputs/gsb_rubric.json').read_bytes ())==ref ['input_hashes']['gsb_rubric.json']
    rubric =read (ROOT /'inputs/gsb_rubric.json');assert SHA (rubric ['SYSTEM'].encode ())==ref ['system_sha256']
    control_rows =[r for r in rows (ROOT /'inputs/control-pairs.jsonl')if r ['pe_cfg']==2 ]
    controls ={r ['sample_id']:r for r in control_rows };assert len (controls )==len (control_rows )==1471 
    complete =read (ROOT /'inputs/COMPLETE.json');proof =read (ROOT /'inputs/PUBLICATION_VERIFIED.json');campaign =read (ROOT /'inputs/generation-campaign.json')
    assert campaign['dit_step']==6000
    assert complete['arms']['3000']['dit_checkpoint'].endswith('/step=6000.ckpt/')
    N =complete ['arms']['3000']['images'];J =2 *N 
    assert complete ['status']==proof ['status']=='PASS'and complete ['inputs']==1464 
    base =campaign ['publication_prefix']+'gsb-vs-qwen21/'
    pairs =[];tasks =[];cohorts ={};labels ={};exclusions ={};valid ={}
    for step in ['3000']:
        done =complete ['arms'][step ];index_path =ROOT /'inputs/generation'/step /'editor/score-index.jsonl'
        images_path =index_path .with_name ('images.manifest.jsonl');excluded_path =ROOT /'inputs/generation'/step /'caption/excluded.jsonl'
        for name ,p in [('editor/score-index.jsonl',index_path ),('editor/images.manifest.jsonl',images_path ),('caption/excluded.jsonl',excluded_path )]:
            assert SHA (p .read_bytes ())==done ['artifacts'][name ]['sha256'],(step ,name )
        generated =rows (index_path );images ={r ['sample_id']:r for r in rows (images_path )};excluded =rows (excluded_path )
        ids ={r ['sample_id']for r in generated }
        assert len (ids )==len (generated )==len (images )==done ['images']==proof ['arms'][step ]['images']
        assert set (images )==ids and not ids &{r ['sample_id']for r in excluded }and len (generated )+len (excluded )==1464 
        assert done ['denoiser_keys']==463 and done ['cfg']==2.5 and done ['steps']==50 and done ['timeshift']==4.6 and done ['weights']=='EMA'
        assert done ['input_order']=='trainorder-v2'and done ['use_qwen_pe']is True and done ['checkpoint_step']==int (step )
        assert proof ['arms'][step ]['all_s3_sha256_match']and proof ['arms'][step ]['all_receipts_match']
        arm =f'pe39c59_qwenpe_c{step }_greedy_loop_retry_dit6k_s6000_ema_cfg25';labels [arm ]=f'Qwen-PE + PE39c59 checkpoint-{step } greedy + rule-triggered retry + DiT6000 EMA CFG2.5 native2K'
        cohorts [arm ]={'count':len (ids ),'sample_ids':sorted (ids )};exclusions [step ]=len (excluded );valid [step ]=len (ids )
        for row in sorted (generated ,key =lambda r :r ['input_index']):
            old =controls [row ['sample_id']];image =images [row ['sample_id']];control =old ['candidates'][CONTROL ]
            assert row ['input_index']==old ['input_index']and row ['request_id']==old ['case_id']==image ['request_id']
            assert row ['original_instruction']==old ['instruction_en']
            assert row ['use_qwen_pe']and row ['instruction_policy']==campaign ['instruction_policy']
            assert SHA (row ['instruction'].encode ())==row ['rewritten_instruction_sha256']
            assert row ['qwen_pe_predictions_sha256']==campaign ['qwen_pe_predictions_sha256']
            assert row ['task']==old ['group']and row ['source_s3_uri']==old ['source']['uri']and row ['source_sha256']==old ['source']['sha256']
            assert row ['original_source_size']==old ['source']['size']
            size =[row ['expected_output_width'],row ['expected_output_height']]
            assert size ==[2 *d for d in control ['size']]
            assert row ['generation_seed']==42 and row ['caption_preparation_version']=='trainorder-v2-native2k-v1'
            assert row ['output_s3_uri']==image ['uri']and row ['output_sha256']==image ['sha256']
            pair ={k :old [k ]for k in ['case_id','sample_id','input_index','group','instruction_en','checkpoints','source']}
            pair .update (pe_arm =arm ,pe_cfg =2.5 ,control_cfg =1 ,checkpoint_step =int (step ),caption_run ='39c59fafbc35',editor_checkpoint_step =6000 ,
            generation_seed =42 ,caption_policy =campaign['sampling']['strategy'],candidates ={arm :dict (uri =image ['uri'],sha256 =image ['sha256'],size =size ),CONTROL :control })
            pair ['pair_id']=SHA (ENC (pair ))[:24 ];pairs .append (pair )
            for order ,first ,second in [('ab',arm ,CONTROL ),('ba',CONTROL ,arm )]:
                task =dict (pair ,order =order ,first =first ,second =second );task ['task_id']=SHA (ENC (task ))[:24 ];tasks .append (task )
    common =set .intersection (*(set (v ['sample_ids'])for v in cohorts .values ()))
    assert sorted (common )==read (ROOT /'inputs/common-output-sample-ids.json')and len (common )==N 
    assert len (pairs )==len ({r ['pair_id']for r in pairs })==N 
    assert len (tasks )==len ({r ['task_id']for r in tasks })==J 
    tasks .sort (key =lambda t :SHA (('seed42:'+t ['task_id']).encode ()))
    for name ,values in [('pairs',pairs ),('tasks',tasks )]:(ROOT /f'inputs/{name }.jsonl').write_bytes (b''.join (ENC (r )for r in values ))
    shards = {str(rank): [] for rank in range(4)}
    for i, pair in enumerate(sorted(pairs, key=lambda r:r['pair_id'])):
        shards[str(i % 4)].append(pair['pair_id'])
    assert len({p for values in shards.values() for p in values}) == N
    (ROOT/'inputs/worker-shards.json').write_bytes(ENC(shards))
    plan =dict (pairs =N ,judgments =J ,common_cases =N ,valid_case_counts =valid ,excluded_case_counts =exclusions ,
    workers=4, concurrency_per_worker=256, total_concurrency=1024, concurrency =1024 ,control ='Frozen Qwen-Image 2.1 CFG1 controls',judge ='gemini-3.8-flash LOW temperature0 AB/BA',
    execution ='Arnold UTP JobSet, four indexed CPU workers with disjoint pair shards',generation ='Qwen-PE rewritten instruction -> PE run39c59fafbc35 c3000 greedy + rule-triggered retry -> DiT6000 EMA463 native2K CFG2.5 steps50 timeshift4.6 seed42')
    (ROOT /'execution-plan.json').write_bytes (ENC (plan ))
    keys =('model','thinking_level','schema','system_sha256','temperature','max_output_tokens','pixel_budget','normalization','tie_rule','double_order','dimensions','dimension_labels','primary_dimension','aggregation','ground_truth','upstream_protocol')
    protocol ={k :ref [k ]for k in keys }
    code =['code/worker.py','driver.py','finalize.py','prepare.py','launch.py','requirements.txt','execution-plan.json','utp_driver.py','merge_workers.py']
    protocol .update (version ='pe39c59-qwenpe-c3000-greedy-loop-retry-dit6000-cfg25-vs-qwen21-gsb-utp4x256-v1',backend ='vertex-gemini',
    workers =4 ,concurrency_per_worker =256 ,total_concurrency =1024 ,pair_storage_concurrency =32 ,
    publication_prefix =base ,output_prefix =base +'judgments/',pe_arms =list (labels ),arm_labels =labels ,
    control_arm =CONTROL ,pe_cfg =2.5 ,control_cfg =1 ,pairs_count =N ,tasks_count =J ,
    unique_cases =len ({r ['sample_id']for r in pairs }),source_cases =1464 ,valid_case_counts =valid ,excluded_case_counts =exclusions ,
    outcome_mapping ={**{arm :'good'for arm in labels },CONTROL :'bad','tie':'similar'},
    cohort =dict (comparisons =cohorts ,all_models_common_count =N ,all_models_common_sample_ids =sorted (common )),
    execution_policy ='UTP 4 indexed CPU workers, 256 requests each, pod AWS identity via Vertex WIF',
    generation_proof =complete ,publication_verification =proof ,
    input_hashes ={str (p .relative_to (ROOT /'inputs')):SHA (p .read_bytes ())for p in sorted ((ROOT /'inputs').rglob ('*'))if p .is_file ()},
    code_hashes ={name :SHA ((ROOT /name ).read_bytes ())for name in code },original_worker_sha256 =ref ['code_hashes']['code/worker.py'],
    worker_changes ='Unchanged from previously validated UTP 4x256 worker; same rubric and scoring functions.')
    assert protocol ['code_hashes']['code/worker.py']=='cd6c24a4ca68de0911d7be79bd2449b60a889d159836b8ed1a576f96c866c244'
    (ROOT /'protocol.json').write_bytes (ENC (protocol ))
    with tarfile .open (ROOT /'bundle.tar.gz','w:gz')as tar :
        for p in [ROOT /'protocol.json',*[ROOT /n for n in code ],*sorted ((ROOT /'inputs').rglob ('*'))]:
            if p .is_file ():tar .add (p ,arcname =str (p .relative_to (ROOT )))
    print (json .dumps (dict (status ='PASS',pairs =len (pairs ),tasks =len (tasks ),common_cases =len (common ),protocol_sha256 =SHA (ENC (protocol )),bundle_sha256 =SHA ((ROOT /'bundle.tar.gz').read_bytes ()),publication_prefix =base )))
if __name__ =='__main__':main ()
