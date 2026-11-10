"""Independent reference integrity, privacy/utility aggregation and missingness contracts."""
import hashlib,json,math,random,statistics
from pathlib import Path

def joint(role,p,u,rule='product'):
    if role not in {'privacy-only','utility-only','dual-role'}:raise ValueError('unknown role')
    for value in (p,u):
        if value is not None and (not math.isfinite(value) or not 0<=value<=1):raise ValueError('score outside [0,1]')
    if role=='privacy-only':return p
    if role=='utility-only':return u
    if p is None or u is None:return None
    if rule=='product':return p*u
    if rule=='minimum':return min(p,u)
    if rule=='harmonic':return 2*p*u/(p+u) if p+u else 0.
    if rule=='privacy_constrained':return u if p==1 else None
    raise ValueError('unknown composite')

def validate(case,codes):
    if not case['source'].strip():raise ValueError('empty source')
    if case['split'] not in {'development','holdout'}:raise ValueError('unknown split')
    seen=set()
    for cue in case['cues']:
        if cue['id'] in seen:raise ValueError('duplicate reference')
        seen.add(cue['id'])
        if not set(cue['labels'])<=codes:raise ValueError('unknown cue')
        evidence=cue['evidence']
        if not evidence or evidence not in case['source']:raise ValueError('ungrounded reference')
        if 'source_span' in cue or ('start' in cue and 'end' in cue):
            start,end=cue.get('source_span',(cue.get('start'),cue.get('end')))
            if not 0<=start<end<=len(case['source']) or case['source'][start:end]!=evidence:raise ValueError('wrong source span')
        if cue['role']!='privacy-only' and (not cue['question'] or not cue['answer']):raise ValueError('missing QA reference')
        if cue['role'] not in {'privacy-only','utility-only','dual-role'}:raise ValueError('unknown role')
        if cue['role']!='utility-only' and not cue['privacy_criteria']:raise ValueError('privacy without threat criteria')
    return True

def bootstrap_case(rows,key,repetitions=2000,seed=42):
    grouped={}
    for r in rows:
        if r.get(key) is not None:grouped.setdefault(r['case_id'],[]).append(r[key])
    vals=[statistics.mean(v) for v in grouped.values()]
    if not vals:return {'mean':None,'ci95':None,'cases':0}
    rng=random.Random(seed);draws=sorted(statistics.mean(rng.choices(vals,k=len(vals))) for _ in range(repetitions))
    return {'mean':statistics.mean(vals),'ci95':[draws[int(.025*repetitions)],draws[int(.975*repetitions)]],'cases':len(vals)}

def verify_freeze(folder):
    folder=Path(folder);m=json.loads((folder/'manifest.json').read_text())
    for name,h in m['files'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=h:raise ValueError('changed frozen reference: '+name)
    return m
