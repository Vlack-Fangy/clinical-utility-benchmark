import json
from pathlib import Path
import pytest
from clinical_benchmark.scoring import cpups,score_instance,score_qa,validate_case,summarize,token_f1


def test_protocol_preserves_taxonomy_including_generated_binding():
    root=Path(__file__).parents[1]
    spec=json.loads((root/'clinical_benchmark/cue_spec.json').read_text())['cues']
    book=json.loads((root/'contextual-anonymizer-codes.json').read_text())
    assert {c['code'] for c in spec}=={c['code'] for g in book['cue_classes'] for c in g['sub_codes']}
    assert len(spec)==41 and len({c['group'] for c in spec})==8
    assert next(c for c in spec if c['code']=='NVB-BOUND')['generated']


def test_composite_cannot_compensate_dual_role_leakage_with_utility():
    assert cpups('dual-role',0,1)==0
    assert cpups('dual-role',1,0)==0
    assert cpups('dual-role',None,1) is None
    assert cpups('privacy-only',1,None)==1
    assert cpups('utility-only',None,.5)==.5
    with pytest.raises(ValueError):cpups('dual-role',3,1)


def cue():
    return {'id':'g','labels':['DIR-GEO','DEM-GEO'],'role':'dual-role',
        'privacy_criteria':[{'kind':'literal_specificity','forbidden':['Shimla']}],
        'utility_atoms':[{'source_evidence':'I experienced discrimination','variants':['I experienced discrimination'],
        'contradictions':['I did not experience discrimination']}]}


def test_joint_outcomes_and_unknown_execution():
    c=cue()
    assert score_instance(c,'In Shimla, I experienced discrimination')['CPUPS']==0
    assert score_instance(c,'I experienced discrimination in a mountainous region')['CPUPS']==1
    assert score_instance(c,'I moved')['CPUPS']==0
    assert score_instance(c,None)['P'] is None
    assert score_instance(c,'I did not experience discrimination')['contradictions']==1
    c['privacy_criteria'][0]['kind']='unknown_contextual_linkage'
    assert score_instance(c,'I experienced discrimination')['CPUPS'] is None


def test_evidence_provenance_and_multi_label_not_duplicate_micro():
    c=cue();c.update(start=0,end=28,evidence='I experienced discrimination')
    case={'source':'I experienced discrimination','cues':[c]}
    validate_case(case,['DIR-GEO','DEM-GEO'])
    c['end']-=1
    with pytest.raises(ValueError):validate_case(case,['DIR-GEO','DEM-GEO'])
    row={**score_instance(c,'I experienced discrimination'),'case_id':'case'}
    assert summarize([row],'CPUPS')['measured_instances']==1
    assert summarize([],'U')['micro_mean'] is None


def test_qa_unsupported_intent_and_partial_preservation():
    q={'id':'q','category':'intent','answer':'no intent','support_variants':['no intent to act'],
       'contradictions':['I intend to act']}
    assert score_qa(q,'I have thoughts of death but no intent to act')['supported']
    assert not score_qa(q,'I have thoughts of death. I intend to act')['supported']
    c=cue();c['utility_atoms'].append({'source_evidence':'severe pain','variants':['severe pain']})
    assert score_instance(c,'I experienced discrimination')['U']==.5
    assert token_f1('no fever','fever')>0
