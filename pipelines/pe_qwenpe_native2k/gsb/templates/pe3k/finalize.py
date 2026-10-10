"""Audit every raw judgment and independently aggregate full and common cohorts."""
from collections import Counter 
from datetime import datetime ,timezone 
import csv 
import hashlib 
import json 
import os 
from pathlib import Path 

ROOT =Path (__file__ ).resolve ().parent 
DATA =Path (os .environ ['CANVA_CLUSTER_STORAGE_PATH'])
ENC =lambda x :(json .dumps (x ,ensure_ascii =False ,sort_keys =True )+'\n').encode ()
SHA =lambda b :hashlib .sha256 (b ).hexdigest ()
read =lambda p :json .loads (p .read_bytes ())
rows =lambda p :[json .loads (x )for x in p .read_bytes ().splitlines ()if x ]
DIMS =('consistency','quality')


def metrics (values ,dimension ):
    c =Counter (r ['dimensions'][dimension ]['outcome']for r in values )
    g ,s ,b =(c [x ]for x in ('good','similar','bad'))
    n =len (values )
    assert g +s +b ==n 
    return dict (n =n ,good =g ,similar =s ,bad =b ,net_win_rate =(g -b )/n if n else None ,
    win_rate_excluding_similar =g /(g +b )if g +b else None ,
    position_disagreements =sum (r ['dimensions'][dimension ]['position_disagreement']for r in values ))


def main ():
    protocol =read (ROOT /'protocol.json')
    N =protocol ['pairs_count'];J =protocol ['tasks_count'];assert J ==2 *N 
    digest =SHA ((ROOT /'protocol.json').read_bytes ())
    for name ,expected in protocol ['input_hashes'].items ():
        assert SHA ((ROOT /'inputs'/name ).read_bytes ())==expected ,name 
    for name ,expected in protocol ['code_hashes'].items ():
        assert SHA ((ROOT /name ).read_bytes ())==expected ,name 
    marker =read (DATA /'ALL_COMPLETE.json')
    assert marker ['status']=='complete'and marker ['protocol_sha256']==digest 
    assert marker ['judgments']==protocol ['tasks_count']==J and marker ['pairs']==protocol ['pairs_count']==N 
    assert SHA ((DATA /'combined.jsonl').read_bytes ())==marker ['combined_sha256']
    pairs ={r ['pair_id']:r for r in rows (ROOT /'inputs/pairs.jsonl')}
    tasks ={r ['task_id']:r for r in rows (ROOT /'inputs/tasks.jsonl')}
    assert len (pairs )==N and len (tasks )==J 
    assert {p .stem for p in (DATA /'results').glob ('*.json')}==set (tasks )
    rubric =read (ROOT /'inputs/gsb_rubric.json')
    orders ={}
    for tid ,task in tasks .items ():
        r =read (DATA /'results'/f'{tid }.json')
        assert r ['status']=='complete'and r ['task']==task and r ['protocol_sha256']==digest 
        cfg =r ['request_config']
        assert cfg ['system_instruction']==rubric ['SYSTEM']
        assert cfg ['temperature']==0 and cfg ['response_mime_type']=='application/json'
        assert cfg ['thinking_config']['thinking_level']=='LOW'
        assert cfg ['max_output_tokens']==protocol ['max_output_tokens']
        assert r ['model_requested']==protocol ['model']
        answer =json .loads (r ['attempts'][-1 ]['response_text'])
        mapping ={'1':task ['first'],'2':task ['second'],'tie':'tie'}
        assert r ['winners']=={d :mapping [answer [d +'_winner']]for d in DIMS }
        assert r ['reasons']=={d :answer [d +'_reason']for d in DIMS }
        for key ,image in [('source',task ['source']),('first',task ['candidates'][task ['first']]),('second',task ['candidates'][task ['second']])]:
            m =r ['judge_'+key ]
            assert m ['original_sha256']==image ['sha256']and m ['original_size']==image ['size']
            assert m ['judge_size'][0 ]*m ['judge_size'][1 ]<=1048576 and m ['mime_type']=='image/webp'
        orders .setdefault (task ['pair_id'],{})[task ['order']]=r 
    combined =rows (DATA /'combined.jsonl')
    assert len (combined )==N and {r ['pair_id']for r in combined }==set (pairs )
    scores =[]
    for result in combined :
        pair =pairs [result ['pair_id']]
        abba =orders [result ['pair_id']]
        assert set (abba )=={'ab','ba'}
        assert result ['status']=='complete'and result ['protocol_sha256']==digest 
        assert all (result [k ]==pair [k ]for k in ('input_index','sample_id','group','pe_arm'))
        assert abba ['ab']['first']==abba ['ba']['second']==pair ['pe_arm']
        assert abba ['ab']['second']==abba ['ba']['first']==protocol ['control_arm']
        for d in DIMS :
            a ,b =abba ['ab']['winners'][d ],abba ['ba']['winners'][d ]
            winner =a if a ==b else 'tie'
            outcome ='similar'if winner =='tie'else ('good'if winner ==pair ['pe_arm']else 'bad')
            assert result ['dimensions'][d ]['winner']==winner and result ['dimensions'][d ]['outcome']==outcome 
            assert result ['dimensions'][d ]['position_disagreement']==(a !=b )
        scores .append (dict (input_index =pair ['input_index'],sample_id =pair ['sample_id'],group =pair ['group'],
        cfg =pair ['pe_cfg'],pe_arm =pair ['pe_arm'],instruction_en =pair ['instruction_en'],checkpoint_step =pair ['checkpoint_step'],caption_run =pair ['caption_run'],
        candidate_uri =pair ['candidates'][pair ['pe_arm']]['uri'],control_uri =pair ['candidates'][protocol ['control_arm']]['uri'],
        source_uri =pair ['source']['uri'],dimensions =result ['dimensions'],
        order_reasons ={order :dict (first =r ['first'],second =r ['second'],reasons =r ['reasons'],record_uri =r ['record_uri'])for order ,r in abba .items ()}))
    scores .sort (key =lambda r :(protocol ['pe_arms'].index (r ['pe_arm']),r ['input_index']))
    common_ids =set (protocol ['cohort']['all_models_common_sample_ids'])
    assert len (common_ids )==protocol ['cohort']['all_models_common_count']==N 
    progress =read (DATA /'progress/worker-0.json')
    assert progress ['concurrency']==protocol ['total_concurrency']==256 
    assert 0 <progress ['peak_active_requests']<=256 
    worker_summary =read (DATA /'summary.json')
    assert SHA ((DATA /'summary.json').read_bytes ())==marker ['summary_sha256']
    assert worker_summary ['complete']and worker_summary ['completed_calls']==J 
    assert worker_summary ['completed_pairs']==N and worker_summary ['failed_calls']==0 
    comparisons ={};common ={}
    def aggregate (cohort ):
        return dict (cases =len (cohort ),metrics ={d :metrics (cohort ,d )for d in DIMS },
        by_task ={g :{d :metrics ([r for r in cohort if r ['group']==g ],d )for d in DIMS }for g in sorted ({r ['group']for r in cohort })})
    for arm in protocol ['pe_arms']:
        cohort =[r for r in scores if r ['pe_arm']==arm ]
        expected =protocol ['cohort']['comparisons'][arm ]
        assert len (cohort )==expected ['count']and {r ['sample_id']for r in cohort }==set (expected ['sample_ids'])
        comparisons [arm ]=dict (checkpoint_step =cohort [0 ]['checkpoint_step'],caption_run =cohort [0 ]['caption_run'],label =protocol ['arm_labels'][arm ],**aggregate (cohort ))
        shared =[r for r in cohort if r ['sample_id']in common_ids ]
        assert len (shared )==N 
        common [arm ]=dict (checkpoint_step =cohort [0 ]['checkpoint_step'],label =protocol ['arm_labels'][arm ],**aggregate (shared ))
        for d in DIMS :
            for key ,value in comparisons [arm ]['metrics'][d ].items ():assert worker_summary ['comparisons'][arm ]['metrics'][d ][key ]==value 
            for key ,value in common [arm ]['metrics'][d ].items ():assert worker_summary ['all_models_common']['comparisons'][arm ][d ][key ]==value 
    normalized ={}
    for pair_orders in orders .values ():
        for row in pair_orders .values ():
            for name in ('source','first','second'):
                im =row ['judge_'+name ];identity =(im ['normalized_sha256'],tuple (im ['judge_size']));key =im ['original_sha256']
                if key in normalized :assert normalized [key ]==identity 
                else :normalized [key ]=identity 
    versions =Counter (o ['model_version']for item in combined for o in item ['orders'])
    summary =dict (status ='complete',judge =protocol ['model'],thinking_level ='LOW',temperature =0 ,
    pairs =N ,judgments =J ,checkpoints =1 ,common_cases =N ,source_cases_per_checkpoint =1464 ,
    valid_case_counts =protocol ['valid_case_counts'],excluded_case_counts =protocol ['excluded_case_counts'],
    protocol_sha256 =digest ,total_concurrency =256 ,peak_active_requests =progress ['peak_active_requests'],
    comparisons =comparisons ,all_models_common =common ,model_versions =dict (versions ),
    completed_utc =datetime .now (timezone .utc ).isoformat (),control ='Frozen Qwen-Image 2.1 CFG1, 40 steps, seed=42+input_index',
    generation ='Qwen-PE rewritten instruction -> PE run 39c59fafbc35 checkpoint 3000; DiT step4000 EMA463; trainorder-v2; CFG2.5; 50 steps; native2K; timeshift4.6; seed42',
    normalization =protocol ['normalization'],tie_rule =protocol ['tie_rule'],publication_prefix =protocol ['publication_prefix'],
    comparison_meaning ='Qwen-PE + PE39c59 c3000 + DiT4000 EMA CFG2.5 versus frozen Qwen controls, judged against original user instructions; only successfully generated cases are scored.')
    (ROOT /'scores.jsonl').write_bytes (b''.join (ENC (r )for r in scores ))
    (ROOT /'summary.json').write_bytes (ENC (summary ))
    with (ROOT /'metrics.csv').open ('w')as f :
        fields =['scope','arm','checkpoint_step','dimension','n','good','similar','bad','net_win_rate','win_rate_excluding_similar','position_disagreements']
        writer =csv .DictWriter (f ,fieldnames =fields );writer .writeheader ()
        for scope ,values in [('all_valid',comparisons ),('common_all_valid',common )]:
            for arm ,value in values .items ():
                for dim ,m in value ['metrics'].items ():writer .writerow (dict (scope =scope ,arm =arm ,checkpoint_step =value ['checkpoint_step'],dimension =dim ,**m ))
    verified =dict (status ='PASS',pairs =N ,judgments =J ,checkpoints =1 ,common_cases =N ,
    protocol_sha256 =digest ,scores_sha256 =SHA ((ROOT /'scores.jsonl').read_bytes ()),summary_sha256 =SHA ((ROOT /'summary.json').read_bytes ()),
    metrics_sha256 =SHA ((ROOT /'metrics.csv').read_bytes ()),raw_judgments_audited =len (orders )*2 ,normalized_images_verified =len (normalized ),
    model_versions =dict (versions ),completed_utc =summary ['completed_utc'])
    (ROOT /'VERIFIED.json').write_bytes (ENC (verified ));print (json .dumps (verified ),flush =True )

if __name__ =='__main__':main ()
