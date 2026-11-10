"""Finite synthetic-fixture correspondence, not population re-identification."""
import hashlib
from .scoring import contains


def candidates(output: str, reference: dict[str, set[str]]) -> dict:
    vocabulary=set().union(*reference.values()) if reference else set()
    observed={value for value in vocabulary if contains(output,value)}
    matched=[key for key,values in reference.items() if observed<=values]
    matched.sort(key=lambda key:hashlib.sha256(('42:'+key).encode()).hexdigest())
    return {'candidates':matched,'observed_criteria':len(observed),'no_evidence':not observed,
        'unmatched':bool(observed) and not matched,'reference_size':len(reference),
        'candidate_reduction':1-len(matched)/len(reference) if reference else None}
