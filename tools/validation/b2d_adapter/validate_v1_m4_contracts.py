"""Check cached M4/M2 datasets and original/enriched resume identities; no Trainer."""
import sys, os, json, hashlib, textwrap
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'repositories/DiffusionDrive'))
sys.dont_write_bytecode=True
os.environ.setdefault('MPLCONFIGDIR','/tmp/e2e_v1_mpl')
os.environ.setdefault('HF_HUB_OFFLINE','1')

from tools.training.diffusiondrive_v1.runtime import model_configuration,bev_provider
from tools.training.diffusiondrive_v1.prepared import prepared_datasets,validate_tensors,atomic_json
from tools.b2d_adapter.errors import AdapterError
root=ROOT;cp=root/'tools/training/diffusiondrive_v1/m4_subset.json';c=json.loads(cp.read_text());model=model_configuration(c)
p=root/'outputs/diffusiondrive_v1/m4_prepared_final/v1_semantic_enriched/m4-v1/accepted_manifest.json'
dsets,info=prepared_datasets(p,c,model)
for split in ('train','validation'):validate_tensors(*dsets[split][0])
errors={}
for name,modified in [('original',dict(c,label_rules='v1_original_7class')),('filter',dict(c,label_rules={'name':'v1_semantic_enriched','actor_floor':'filter-supported-road'}))]:
 try:prepared_datasets(p,modified,model)
 except AdapterError as e:errors[name]=e.code
 else:raise AssertionError('mismatched labels admitted')
assert all(v=='prepared_contract_mismatch' for v in errors.values())
atomic_json(root/'analysis/diffusiondrive_b2d_architecture/v1_validation/m4/prepared_integration.json',{'status':'pass','manifest':info,'different_rules_rejected':errors,'original_model_tensor_contract_unchanged':True,'training_executed':False})
print(info,errors)

from tools.b2d_adapter.v1_builders import validate_v1_config,v1_native_config
from tools.training.diffusiondrive_v1.runtime import model_configuration,bev_provider
from tools.training.diffusiondrive_v1.prepared import prepared_datasets,atomic_json
root=ROOT;source=(root/'tools/training/diffusiondrive_v1/fit.py').read_text();begin=source.index("        semantic={'data':");end=source.index("        report['contract_sha256']",begin)
code=compile(textwrap.dedent(source[begin:end]),'actual fit semantic block','exec')
c=json.loads((root/'tools/training/diffusiondrive_v1/m2_subset.json').read_text());model=model_configuration(c)
report=json.loads((root/'outputs/diffusiondrive_v1/m2_fit_checked/fit_result.json').read_text());_,identity=prepared_datasets(root/c['prepared_manifest'],c,model)
base={'hashlib':hashlib,'json':json,'ROOT':root,'model_config':model,'validate_v1_config':validate_v1_config,'v1_native_config':v1_native_config,'initialization':report['initialization'],'prepared_identity':identity,'config':c,'provider':None,'CommittedSampler':__import__('tools.training.diffusiondrive_v1.sampling',fromlist=['CommittedSampler']).CommittedSampler}
exec(code,base);assert base['contract_hash']==report['contract_sha256'],(base['contract_hash'],report['contract_sha256'])
contracts={}
for rule in ('v1_original_7class','v1_semantic_enriched'):
 scope=dict(base,config=dict(c,label_rules=rule),provider=bev_provider(dict(c,label_rules=rule)),prepared_identity=None)
 exec(code,scope);contracts[rule]=scope['contract_hash']
assert len(set(contracts.values()))==2
atomic_json(root/'analysis/diffusiondrive_b2d_architecture/v1_validation/m4/resume_contract.json',{'status':'pass','historical_prepared_resume_contract_unchanged':base['contract_hash'],'raw_label_rule_contracts':contracts,'training_executed':False})
print('historical contract retained; raw original/enriched resume contracts distinct')
