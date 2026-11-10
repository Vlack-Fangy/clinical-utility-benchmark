"""Standalone frozen-reference scoring; neither detections nor D/M are gold.

Deterministic evidence correspondence is a conservative proxy, not semantic QA.
Unknown privacy criteria stay unmeasured. A composite never clears a release.
"""
from collections import Counter,defaultdict
from dataclasses import dataclass
import math,random,re,statistics


def normalize(text):
    return ' '.join(re.findall(r"\w+|[^\w\s]",text.casefold(),re.UNICODE))


def contains(text,needle):
    return (' '+normalize(needle)+' ') in (' '+normalize(text)+' ')


def cpups(role,p,u):
    if role not in ('privacy-only','utility-only','dual-role'):raise ValueError('invalid role')
    for value in (p,u):
        if value is not None and (not math.isfinite(value) or not 0<=value<=1):raise ValueError('adequacy must be in [0,1]')
    if role=='privacy-only':return p
    if role=='utility-only':return u
    return None if p is None or u is None else p*u


def validate_case(case,codes):
    source=case['source'];ids=set()
    for cue in case['cues']:
        if cue['id'] in ids:raise ValueError('duplicate reference instance')
        ids.add(cue['id'])
        if not cue['labels'] or set(cue['labels'])-set(codes):raise ValueError('unknown cue label')
        if not 0<=cue['start']<cue['end']<=len(source) or source[cue['start']:cue['end']]!=cue['evidence']:raise ValueError('incorrect source evidence')
        if cue['role'] not in ('privacy-only','utility-only','dual-role'):raise ValueError('invalid role')
        for atom in cue.get('utility_atoms',[]):
            if not contains(source,atom['source_evidence']):raise ValueError('ungrounded clinical fact')
            if not atom.get('variants'):raise ValueError('empty accepted evidence variants')
    for q in case.get('questions',[]):
        if not contains(source,q['evidence']):raise ValueError('ungrounded question')


def score_instance(cue,output):
    if output is None:return {'id':cue['id'],'labels':cue['labels'],'role':cue['role'],'P':None,'U':None,'CPUPS':None,'status':'execution_unavailable'}
    privacy=cue.get('privacy_criteria',[]);atoms=cue.get('utility_atoms',[])
    # Removal of predeclared literal specifics is only the declared finite attack.
    retained=[any(contains(output,v) for v in c['forbidden']) for c in privacy]
    p=(1-sum(retained)/len(retained)) if retained and all(c['kind']=='literal_specificity' for c in privacy) else None
    contradiction=[any(contains(output,v) for v in a.get('contradictions',[])) for a in atoms]
    supported=[any(contains(output,v) for v in a['variants']) and not contradiction[i] for i,a in enumerate(atoms)]
    u=sum(supported)/len(atoms) if atoms else None
    return {'id':cue['id'],'labels':cue['labels'],'role':cue['role'],'P':p,'U':u,'CPUPS':cpups(cue['role'],p,u),
        'privacy_criteria':len(privacy),'retained_criteria':sum(retained),'utility_atoms':len(atoms),'supported_atoms':sum(supported),
        'contradictions':sum(contradiction),'status':'measured','privacy_veto':any(retained),
        'unnecessary_removal_proxy':bool(atoms) and not all(supported),
        'generalization_correctness':None,'unsupported_additions_global':None}


def score_qa(question,output):
    if output is None:return {'id':question['id'],'status':'execution_unavailable','supported':None}
    bad=any(contains(output,s) for s in question.get('contradictions',[]))
    support=any(contains(output,s) for s in question['support_variants']) and not bad
    return {'id':question['id'],'category':question['category'],'status':'measured','supported':support,
        'detected_unsupported_conclusion':bad,'answer':question['answer'] if support else None,
        'metric':'frozen evidence answer-support proxy; not independent QA-model accuracy'}


def bootstrap(values,seed=42,n=1000):
    values=list(values)
    if not values:return {'n':0,'mean':None,'ci95':None}
    rng=random.Random(seed);samples=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(n))
    return {'n':len(values),'mean':statistics.mean(values),'ci95':[samples[int(.025*n)],samples[int(.975*n)]],'unit':'case means','seed':seed,'resamples':n}


def summarize(rows,key):
    by_case=defaultdict(list)
    for r in rows:
        if r.get(key) is not None:by_case[r['case_id']].append(r[key])
    values=[r[key] for r in rows if r.get(key) is not None]
    return {'planned_instances':len(rows),'measured_instances':len(values),'micro_mean':statistics.mean(values) if values else None,
        'case_macro':bootstrap([statistics.mean(v) for v in by_case.values()]),
        'coverage_status':'insufficient' if len(by_case)<5 else 'pilot; not validated'}


def token_f1(source,output):
    if output is None:return None
    a=Counter(re.findall(r'\w+',source.casefold()));b=Counter(re.findall(r'\w+',output.casefold()));tp=sum((a&b).values())
    return 2*tp/(sum(a.values())+sum(b.values())) if a or b else None
