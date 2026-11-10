"""Freeze new authored minimal pairs before evaluating any metric.
No anonymizer imports or predictions. Single-author automated drafting, not clinical gold.
"""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
# family, source, valid paraphrase/generalization, clinically altered version, label, forbidden
PAIRS=[
('negation','The patient denies hearing voices while awake.','While awake, the patient reports no voices.','The patient hears voices while awake.','KEEP-SYMPT',[]),
('negation','There was no chest pain during the panic episode.','Chest pain was absent throughout the panic episode.','Chest pain occurred during the panic episode.','KEEP-SYMPT',[]),
('agency','During the argument, the father pushed the patient; the patient did not push him.','The patient was pushed by the father in the argument and did not push back.','During the argument, the patient pushed the father; the father did not push the patient.','KEEP-REL',[]),
('agency','The colleague threatened the patient after work.','After work the patient received a threat from the colleague.','The patient threatened the colleague after work.','KEEP-REL',[]),
('chronology','The nightmares began before the move, and treatment started after the move.','Nightmares preceded relocation; treatment followed relocation.','The nightmares began after the move, and treatment started before the move.','KEEP-SYMPT',[]),
('chronology','Nausea started two days after the medication dose increased.','Two days following the dose increase, nausea began.','Nausea started two days before the medication dose increased.','KEEP-SYMPT',[]),
('severity','The headaches became severe after initially being mild.','Initially mild headaches subsequently became intense.','The headaches became mild after initially being severe.','KEEP-AFFECT',[]),
('severity','Panic episodes decreased from daily to once a week.','Panic attacks now occur weekly rather than every day.','Panic episodes increased from weekly to daily.','KEEP-SYMPT',[]),
('dose','The clinician reduced sertraline from 100 mg to 50 mg daily.','The daily sertraline dose was lowered by the clinician from one hundred to fifty milligrams.','The clinician increased sertraline from 50 mg to 100 mg daily.','KEEP-TECH',[]),
('dose','The prescription is propranolol 10 mg twice daily, not 100 mg.','Propranolol is prescribed at ten milligrams two times a day, rather than one hundred milligrams.','The prescription is propranolol 100 mg twice daily, not 10 mg.','KEEP-TECH',[]),
('relationship','The aunt provides childcare; the mother cannot provide childcare.','Childcare is provided by the aunt because the mother is unable to provide it.','The mother provides childcare; the aunt cannot provide childcare.','KEEP-REL',[]),
('relationship','The patient feels safe with the sister but unsafe with the partner.','The sister feels safe to the patient; the partner does not.','The patient feels safe with the partner but unsafe with the sister.','KEEP-REL',[]),
('coreference','The patient has two cousins. The older cousin borrowed money, and the younger cousin repaid it.','Of two cousins, the older one took the loan and the younger one paid it back.','The patient has two cousins. The younger cousin borrowed money, and the older cousin repaid it.','KEEP-REL',[]),
('coreference','The nurse and the patient spoke. The nurse, not the patient, called the psychiatrist.','After their conversation, it was the nurse who phoned the psychiatrist, not the patient.','The nurse and the patient spoke. The patient, not the nurse, called the psychiatrist.','KEEP-REL',[]),
('uncertainty','The patient is unsure whether the new job contributed to insomnia.','A contribution of the new job to insomnia remains uncertain to the patient.','The patient is certain the new job caused insomnia.','KEEP-SYMPT',[]),
('uncertainty','The clinician has not established whether the shaking is medication-related.','Whether medication explains the shaking remains undetermined by the clinician.','The clinician has established that the shaking is medication-related.','KEEP-SYMPT',[]),
('intent','The patient reports thoughts of death but denies a plan or intent to act.','Death-related thoughts are present, with neither an action plan nor intent reported.','The patient reports thoughts of death with a plan and intent to act.','KEEP-SYMPT',[]),
('intent','The patient denies current suicidal thoughts but reports an attempt several years ago.','There is a past suicide attempt years earlier; current suicidal thinking is denied.','The patient reports current suicidal thoughts but denies any past attempt.','KEEP-SYMPT',[]),
('spatial','During the assault, the patient hid behind a tree and saw the attacker pass; whether the attacker saw the patient is not established.','Concealed behind a tree during the assault, the patient watched the attacker pass; the attacker noticing the patient remains unknown.','During the assault, the patient hid behind a tree and saw the attacker pass; the attacker definitely saw the patient.','KEEP-REL',[]),
('spatial','The patient waited inside the locked room while the threatening visitor remained outside.','A locked door separated the patient inside the room from the threatening visitor outside.','The threatening visitor waited inside the locked room while the patient remained outside.','KEEP-REL',[]),
('deletion','The patient reports breathlessness when climbing stairs and no breathlessness at rest.','Climbing stairs causes reported breathlessness, which is absent at rest.','The patient reports breathlessness at rest and no breathlessness when climbing stairs.','KEEP-SYMPT',[]),
('deletion','The patient stopped attending therapy because transport became unaffordable, not because symptoms improved.','Unaffordable travel, rather than symptom improvement, led the patient to stop therapy attendance.','The patient stopped attending therapy because symptoms improved, not because transport became unaffordable.','KEEP-TECH',[]),
('geography','The patient travels from Shimla; the steep uphill route to treatment worsens breathlessness.','The patient travels from a hill town; the steep uphill route to treatment worsens breathlessness.','The patient travels from a hill town; the steep uphill route to treatment improves breathlessness.','DEM-GEO',['Shimla']),
('geography','After moving from Pune, the patient experienced discrimination about regional background and became reluctant to attend the group.','After moving from another city, the patient experienced regional discrimination and grew reluctant to attend the group.','After moving from another city, the patient experienced regional acceptance and grew eager to attend the group.','DEM-GEO',['Pune']),
('rare_event','The patient survived the 2021 Bhandara bridge collapse and now avoids bridges because they evoke memories of the event.','The patient survived a bridge collapse and now avoids bridges because they evoke memories of the event.','The patient survived a bridge collapse and now seeks out bridges because they evoke pleasant memories of the event.','NAR-RARE',['2021 Bhandara bridge collapse','Bhandara']),
('rare_event','After receiving the fictional Cedar Award for emergency rescue, the patient felt guilt about the person they could not save.','After receiving a rescue award, the patient felt guilt about the person they could not save.','After receiving a rescue award, the patient felt pride about the person they could not save.','NAR-RARE',['Cedar Award']),
('cumulative','In session one, the patient described being the only night ferryman in Kesar village. In session two, the patient explained that isolation at work worsened low mood.','In session one, the patient described isolated night transport work in a village. In session two, the patient explained that isolation at work worsened low mood.','In session one, the patient described isolated night transport work in a village. In session two, the patient explained that isolation at work improved low mood.','DEM-OCC',['only night ferryman','Kesar']),
('cumulative','The patient works at the fictional Blue Lantern Observatory and runs its sole overnight telescope shift. The isolation makes panic harder to manage.','The patient works alone overnight at a scientific facility. The isolation makes panic harder to manage.','The patient works alone overnight at a scientific facility. The isolation makes panic easier to manage.','DEM-OCC',['Blue Lantern Observatory','sole overnight telescope'])]


def build():
 jobs=[];truth=[];sources=[]
 for i,(family,source,para,bad,label,forbidden) in enumerate(PAIRS):
  split='development' if i%2==0 else 'holdout';cid=f'{family}-{split}'
  # Distinct coherent visit frames; identical within each minimal-pair family.
  background=('At follow-up, the patient and clinician reviewed the current concern. The clinician asked for clarification rather than assuming a diagnosis.' if i%2==0 else 'At the next visit, the patient discussed what had changed since the previous appointment. A follow-up conversation was arranged.')
  text=background+' '+source
  ref={'id':'fact-1','role':'dual-role' if forbidden else 'utility-only','labels':[label]+(['DIR-GEO'] if family=='geography' else []),'evidence':source,'source_span':[len(background)+1,len(text)],'question':'What does the source establish about '+family.replace('_',' ')+'? Preserve actors, polarity, chronology, intensity and uncertainty without requiring exact identifying details.','answer':source,'privacy_criteria':[{'forbidden':forbidden}] if forbidden else [],'critical':True,'annotation_status':'single-author automated drafting; source-grounded; no clinician adjudication','session_id':'two_sessions' if family=='cumulative' else 's1','speaker':'reported patient','uncertainty':'author-defined intervention truth; external validity unestablished'}
  sources.append({'id':cid,'split':split,'stratum':'authored_metric_redteam','language':'en','source':text,'cues':[ref],'license':'new synthetic research fixture; no source patient','family':family})
  for variant,body,u in [('unchanged',source,1),('paraphrase',para,1),('altered',bad,0),('deleted','The clinical detail is unavailable.',0)]:
   jid=cid+'-'+variant
   jobs.append({'id':jid,'case_id':cid,'family':family,'method':variant,'stratum':'authored_metric_redteam','split':split,'source':text,'output':background+' '+body,'cues':[ref]})
   truth.append({'id':jid,'case_id':cid,'family':family,'variant':variant,'split':split,'U':u,'P':(0 if variant=='unchanged' else 1) if forbidden else None})
 return sources,jobs,truth

if __name__=='__main__':
 sources,jobs,truth=build();out=ROOT/'data';out.mkdir(exist_ok=True)
 for name,data in [('probes.json',sources),('judge_inputs.json',jobs),('truth.json',truth)]:
  p=out/name
  if p.exists():raise SystemExit('Frozen file exists: '+str(p))
  p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 manifest={'version':'metric-redteam-v1','seed':42,'source_cases':len(sources),'jobs':len(jobs),'splits':{'development':14,'holdout':14},'authoring':'Codex-assisted single-author source/perturbation drafting before scoring; not independently clinician adjudicated','threshold_selection':'development only; maximize balanced accuracy, ties nearest 0.5 then lowest threshold','ci':'case-cluster bootstrap 2000 samples seed42','privacy':'declared literal and semantic specificity only; no real-person attack','models':['Gemma 4 31B IT frozen offline rubric','bert-base-uncased layer9 BERTScore F1','cross-encoder/nli-deberta-v3-base entailment'],'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.json')}}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print({'sources':len(sources),'jobs':len(jobs),'manifest':hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()})
