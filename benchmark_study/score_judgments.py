"""Portable external-judge adapter. No production imports or model dependencies."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from clinical_benchmark.semantic import validate_judgment

def score(job,judgment):
    expected=[r['id'] for r in job['cues']]
    answers=judgment.get('answers',[])
    if sorted(a.get('id','') for a in answers)!=sorted(expected):raise ValueError('reference coverage mismatch')
    indexed={a['id']:a for a in answers}
    return [validate_judgment(r,job['output'],indexed[r['id']]) for r in job['cues']]

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--judgments',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise ValueError('fresh output required')
 jobs=json.loads(a.inputs.read_text());judgments=json.loads(a.judgments.read_text());rows=[]
 for j in jobs:
  try:
   if j['id'] not in judgments:raise ValueError('missing judgment')
   rows.append({'id':j['id'],'status':'scored','scores':score(j,judgments[j['id']])})
  except (ValueError,KeyError,TypeError) as e:rows.append({'id':j['id'],'status':'failed','error':str(e),'scores':[]})
 a.out.write_text(json.dumps(rows,indent=2)+'\n')
