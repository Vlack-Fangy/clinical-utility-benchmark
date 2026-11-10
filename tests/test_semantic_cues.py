import pytest
from clinical_benchmark.semantic import validate_judgment


def reference(role='dual-role'):
    return {'id':'f1','role':role,'privacy_criteria':[{'forbidden':['Asha Rao']}],
        'evidence':'Asha Rao is supported by her sister.', 'answer':'Her sister supports her.',
        'contradictions':['Her sister does not support her.']}


def judgment(**changes):
    row={'id':'f1','privacy':'protected','utility':'preserved',
         'privacy_evidence':'','utility_evidence':'Her sister supports her.'}
    return dict(row,**changes)


def test_paraphrase_receives_full_credit_with_contiguous_output_evidence():
    row=validate_judgment(reference(), 'Her sister supports her.', judgment())
    assert (row['P'],row['U'],row['CPUPS'])==(1,1,1)


def test_literal_leakage_vetoes_semantic_privacy_and_composite():
    row=validate_judgment(reference(), 'Asha Rao. Her sister supports her.', judgment())
    assert row['P']==0 and row['CPUPS']==0 and row['leakage_veto']


def test_fabricated_support_uncertain_and_partial_are_not_full_preservation():
    assert validate_judgment(reference(), 'No clinical detail.', judgment())['U'] is None
    assert validate_judgment(reference(), 'Her sister supports her.', judgment(utility='uncertain'))['U'] is None
    assert validate_judgment(reference(), 'Her sister supports her.', judgment(utility='partial'))['U']==.5


def test_negation_overrides_positive_support_without_erasing_privacy_result():
    row=validate_judgment(reference(), 'Her sister supports her. Her sister does not support her.', judgment())
    assert row['U']==0 and row['P']==1


def test_missing_or_wrong_reference_id_is_rejected():
    with pytest.raises(ValueError):validate_judgment(reference(),'text',judgment(id='wrong'))
