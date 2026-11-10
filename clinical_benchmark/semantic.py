"""Frozen-reference semantic rubric with bounded deterministic safeguards.

No anonymizer imports. A judge is external and evaluated on saved outputs only.
Evidence validation is necessary, not sufficient, for a correct semantic judgment.
"""
from .evidence import quote_evidence_span
from .scoring import contains, cpups


def validate_judgment(reference: dict, output: str, judgment: dict) -> dict:
    if judgment.get('id') != reference['id']:
        raise ValueError('reference identity mismatch')
    role = reference['role']
    p_status, u_status = judgment.get('privacy'), judgment.get('utility')
    if p_status not in {'protected','retained','uncertain','not_applicable'} or u_status not in {'preserved','partial','lost','altered','uncertain','not_applicable'}:
        raise ValueError('unknown rubric status')
    errors = []
    retained = [any(contains(output,v) for v in c['forbidden']) for c in reference.get('privacy_criteria',[])]
    p = None
    if role != 'utility-only':
        p = {'protected':1.0,'retained':0.0}.get(p_status)
        if p_status == 'retained' and not quote_evidence_span(output,judgment.get('privacy_evidence','')):
            p = None
            errors.append('ungrounded_privacy_evidence')
        if any(retained):
            p = 0.0
    u = None
    if role != 'privacy-only':
        u = {'preserved':1.0,'partial':.5,'lost':0.0,'altered':0.0}.get(u_status)
        if u_status in {'preserved','partial','altered'} and not quote_evidence_span(output,judgment.get('utility_evidence','')):
            u = None
            errors.append('ungrounded_utility_evidence')
        if any(contains(output,c) for c in reference.get('contradictions',[])):
            u = 0.0
            errors.append('declared_contradiction_present')
        elif u_status in {'lost','altered'} and reference['evidence'] in output:
            u = None
            errors.append('contradicts_verbatim_reference')
    return {'id':reference['id'],'P':p,'U':u,'CPUPS':cpups(role,p,u),
        'role':role,'privacy_status':p_status,'utility_status':u_status,
        'retained_criteria':sum(retained),'privacy_criteria':len(retained),
        'leakage_veto':any(retained) or p==0, 'validation_errors':errors,
        'utility_span':quote_evidence_span(output,judgment.get('utility_evidence','')),
        'privacy_span':quote_evidence_span(output,judgment.get('privacy_evidence',''))}
