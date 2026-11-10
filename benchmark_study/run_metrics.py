"""Official ROUGE/BERTScore and generic NLI, CPU only, fixed inputs, no gold-aware scoring."""

import os
from pathlib import Path
MODELS_ROOT = Path(os.environ.get("CLINICAL_BENCHMARK_MODELS_ROOT", "./models")).expanduser()
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TOKENIZERS_PARALLELISM']='false'
import hashlib,json,time,sys
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from bert_score import BERTScorer
from rouge_score import rouge_scorer
ROOT=Path(__file__).resolve().parent
(ROOT/'results').mkdir(parents=True, exist_ok=True)

def main():
 torch.set_num_threads(4);torch.manual_seed(42)
 jobs=json.loads((ROOT/'data/judge_inputs.json').read_text())
 out=ROOT/'results/metrics.json'
 if out.exists():raise ValueError('do not overwrite frozen results')
 start=time.monotonic();rouge=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=False)
 scorer=BERTScorer(model_type=str(MODELS_ROOT / 'bert-base-uncased'),num_layers=9,device='cpu',batch_size=8,idf=False,rescale_with_baseline=False)
 _,_,f=scorer.score([j['output'] for j in jobs],[j['source'] for j in jobs]);bert_seconds=time.monotonic()-start
 rows=[]
 for j,v in zip(jobs,f.tolist()):
  rows.append({'id':j['id'],'rougeL':rouge.score(j['source'],j['output'])['rougeL'].fmeasure,'bert_score_F1':v,'exact_reference':float(j['cues'][0]['evidence'] in j['output'])})
 del scorer
 modelpath=Path(str(MODELS_ROOT / 'cross-encoder-nli-deberta-v3-base'))
 tokenizer=AutoTokenizer.from_pretrained(modelpath,local_files_only=True)
 model=AutoModelForSequenceClassification.from_pretrained(modelpath,local_files_only=True).eval()
 assert model.config.id2label[1]=='entailment'
 start=time.monotonic()
 for offset in range(0,len(jobs),8):
  batch=jobs[offset:offset+8]
  encoded=tokenizer([j['output'] for j in batch],[j['cues'][0]['evidence'] for j in batch],padding=True,truncation=False,return_tensors='pt')
  if encoded['input_ids'].shape[1]>512:raise ValueError('NLI context too long; no silent truncation')
  with torch.no_grad():probs=model(**encoded).logits.softmax(-1)
  for row,p in zip(rows[offset:offset+8],probs.tolist()):row.update(nli_entailment=p[1],nli_contradiction=p[0],nli_neutral=p[2])
 out.write_text(json.dumps(rows,indent=2)+'\n')
 manifest={'n':len(jobs),'bert_seconds':bert_seconds,'nli_seconds':time.monotonic()-start,'device':'CPU','threads':4,'torch':torch.__version__,'models':{}}
 for path in [Path(str(MODELS_ROOT / 'bert-base-uncased')),modelpath]:
  manifest['models'][path.name]={}
  for p in sorted(path.glob('*')):
   if p.name in ['config.json','tokenizer.json','tokenizer_config.json','vocab.txt','spm.model','model.safetensors','pytorch_model.bin']:
    h=hashlib.sha256()
    with p.open('rb') as stream:
     for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    manifest['models'][path.name][p.name]=h.hexdigest()
 (ROOT/'results/metric_runtime.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'n':len(rows),'bert_seconds':bert_seconds,'nli_seconds':manifest['nli_seconds']}))
if __name__=='__main__':main()
