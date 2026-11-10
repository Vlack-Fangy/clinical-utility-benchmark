
import os
from pathlib import Path
MODELS_ROOT = Path(os.environ.get("CLINICAL_BENCHMARK_MODELS_ROOT", "./models")).expanduser()
import ast,copy,json
from pathlib import Path
import pytest
from benchmark_study.protocol import joint,validate,verify_freeze,bootstrap_case
ROOT=Path(__file__).resolve().parents[1]
CODES={c['code'] for c in json.loads((ROOT.parent/'clinical_benchmark/cue_spec.json').read_text())['cues']}
CASES=json.loads((ROOT/'data/probes.json').read_text())

def test_freeze():assert verify_freeze(ROOT/'data')['jobs']==112
@pytest.mark.parametrize('case',CASES,ids=lambda c:c['id'])
def test_source_integrity(case):assert validate(case,CODES)
@pytest.mark.parametrize('role,p,u,expected',[('privacy-only',0,1,0),('utility-only',0,1,1),('dual-role',1,1,1),('dual-role',0,1,0),('dual-role',1,0,0),('dual-role',None,1,None),('dual-role',1,None,None),('dual-role',.5,.5,.25)])
def test_joint(role,p,u,expected):assert joint(role,p,u)==expected

def test_malformed_evidence():
 c=copy.deepcopy(CASES[0]);c['cues'][0]['source_span']=[0,1]
 with pytest.raises(ValueError):validate(c,CODES)
def test_duplicate():
 c=copy.deepcopy(CASES[0]);c['cues']*=2
 with pytest.raises(ValueError):validate(c,CODES)
def test_unknown_not_zero():assert bootstrap_case([{'case_id':'a','score':None}],'score')['mean'] is None
def test_case_clusters():assert bootstrap_case([{'case_id':'a','score':1}]*100+[{'case_id':'b','score':0}],'score')['mean']==.5
def test_no_anonymizer_dependency():
 for f in [ROOT/'protocol.py',ROOT/'build_probes.py',ROOT/'run_metrics.py']:
  tree=ast.parse(f.read_text())
  assert all(not (isinstance(n,ast.ImportFrom) and (n.module or '').startswith('anon')) for n in ast.walk(tree))
def test_perturbations_and_splits():
 jobs=json.loads((ROOT/'data/judge_inputs.json').read_text());truth=json.loads((ROOT/'data/truth.json').read_text())
 assert len({j['id'] for j in jobs})==112
 assert {j['id'] for j in jobs}=={t['id'] for t in truth}
 assert {c['source'] for c in CASES if c['split']=='development'}.isdisjoint(c['source'] for c in CASES if c['split']=='holdout')

def test_label_blind_semantic_contract():
 from clinical_benchmark.semantic import validate_judgment
 r={'id':'x','role':'utility-only','labels':['KEEP-SYMPT'],'evidence':'No fever.','privacy_criteria':[]}
 j={'id':'x','privacy':'not_applicable','utility':'preserved','utility_evidence':'Fever is absent.'}
 before=validate_judgment(r,'Fever is absent.',j)
 r['labels']=['KEEP-REL','NVB-SILENCE']
 assert validate_judgment(r,'Fever is absent.',j)==before

def test_tokenizer_context_regression():
 pytest.importorskip('transformers')
 if not Path(str(MODELS_ROOT / 'deberta-large-mnli')).exists():pytest.skip('local tokenizer checkpoint unavailable')
 from transformers import AutoTokenizer
 t=AutoTokenizer.from_pretrained(str(MODELS_ROOT / 'deberta-large-mnli'),local_files_only=True)
 t.model_max_length=512
 jobs=json.loads((ROOT/'data/judge_inputs.json').read_text())
 assert max(len(t.encode(j['output'],truncation=False)) for j in jobs)<512
 assert len(t.encode('A short clinical sentence.',truncation=True))<512

@pytest.mark.parametrize('case',json.loads((ROOT/'data/legacy/references.json').read_text()),ids=lambda c:c['id'])
def test_legacy_reference_integrity(case):assert validate(case,CODES)

def test_external_judge_missing_reference():
 from benchmark_study.score_judgments import score
 with pytest.raises(ValueError):score({'cues':[{'id':'x'}]}, {'answers':[]})
