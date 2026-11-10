"""Prespecified supplementary BERTScore checkpoint (official recommended DeBERTa layer18)."""

import os
from pathlib import Path
MODELS_ROOT = Path(os.environ.get("CLINICAL_BENCHMARK_MODELS_ROOT", "./models")).expanduser()
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TOKENIZERS_PARALLELISM']='false'
import json,time,hashlib
from pathlib import Path
import torch
from bert_score import BERTScorer
ROOT=Path(__file__).resolve().parent
(ROOT/'results').mkdir(parents=True, exist_ok=True)
if __name__=='__main__':
 torch.set_num_threads(4);torch.manual_seed(42)
 jobs=json.loads((ROOT/'data/judge_inputs.json').read_text());out=ROOT/'results/bert_deberta.json'
 if out.exists():raise ValueError('fresh output required')
 start=time.monotonic()
 modelpath=Path(str(MODELS_ROOT / 'deberta-large-mnli'))
 scorer=BERTScorer(model_type=str(modelpath),num_layers=18,device='cpu',batch_size=4,idf=False,rescale_with_baseline=False)
 scorer._tokenizer.model_max_length=512  # local tokenizer sentinel exceeds Rust integer range
 lengths=[len(scorer._tokenizer.encode(t, truncation=False)) for j in jobs for t in (j['source'],j['output'])]
 if max(lengths)>512:raise ValueError('input exceeds model context; no silent truncation')
 _,_,values=scorer.score([j['output'] for j in jobs],[j['source'] for j in jobs])
 out.write_text(json.dumps([{'id':j['id'],'bert_deberta_F1':v} for j,v in zip(jobs,values.tolist())],indent=2)+'\n')
 hashes={}
 for p in modelpath.glob('*'):
  if p.name in ['config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','model.safetensors','pytorch_model.bin']:
   h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
   hashes[p.name]=h.hexdigest()
 (ROOT/'results/bert_deberta_runtime.json').write_text(json.dumps({'n':len(jobs),'seconds':time.monotonic()-start,'model_hashes':hashes,'library':'bert-score0.3.13','layer':18,'device':'CPU','rescale':False,'idf':False},indent=2))
 print('Completed',len(jobs),'in',time.monotonic()-start,'seconds')
