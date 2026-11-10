# Clinical Utility Benchmark

Independent research tools for checking what clinical information survives text
anonymization. The scorer compares a candidate output with source-grounded
references, keeping clinical utility, privacy criteria and missing evidence
separate. It does not use anonymizer predictions as ground truth.

## Included

- `clinical_benchmark/`: reference scoring, evidence validation, semantic-judge
  validation and a 41-cue taxonomy.
- `benchmark_study/protocol.py`: input validation, joint scores and case-level
  bootstrap aggregation.
- `benchmark_study/data/`: 28 authored synthetic sources, four interventions
  each, frozen into 14 development and 14 holdout cases (112 jobs).
- `benchmark_study/data/legacy/`: 16 authored synthetic reference cases and four
  licensed ACI-Bench excerpts, with provenance and attribution.
- `benchmark_study/run_metrics.py`: ROUGE, BERTScore and NLI evaluation.
- `benchmark_study/score_judgments.py`: validation of externally supplied judgments.

No source interviews, patient records, identity mappings, anonymizer internals
or private inference traces are included. Historical analysis/report generators
that depend on private run directories are outside this standalone release.

## Quick start: CPU checks

Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -c "from benchmark_study.protocol import verify_freeze; print(verify_freeze('benchmark_study/data'))"
```

The core checks require no model download. One optional tokenizer regression
runs only when Transformers and its local checkpoint are available.

## Optional model-based metrics

```bash
python -m pip install -r requirements.txt
export CLINICAL_BENCHMARK_MODELS_ROOT="$HOME/models"
python benchmark_study/run_metrics.py
python benchmark_study/run_stronger_metric.py
```

Place `bert-base-uncased`, `cross-encoder-nli-deberta-v3-base` and
`deberta-large-mnli` checkpoint folders under the configured root. These runners
use CPU, record model hashes and reject existing result files. Run each fresh
replication in a separate checkout; do not overwrite frozen data or prior runs.

For judgments obtained separately using `benchmark_study/evaluator_prompt.txt`:

```bash
mkdir -p benchmark_study/results
python benchmark_study/score_judgments.py \
  --inputs benchmark_study/data/judge_inputs.json \
  --judgments /path/to/judgments.json \
  --out benchmark_study/results/judgment_scores.json
```

Judgments are a JSON object keyed by job ID, each containing an `answers` list.
Cue IDs must match the job exactly. Missing or malformed judgments are explicit
failures. Consult `clinical_benchmark/semantic.py` for field validation.

## Interpretation and reproducibility

Synthetic fixtures were drafted with AI assistance and have no independent
clinician adjudication. The reused legacy holdout is retrospective, not a new
blind evaluation. Use development data for threshold selection and preserve
holdout separation. Unknown scores are not zero. These measures do not establish
clinical validity, legal compliance or a guarantee of anonymity.

The frozen manifest binds source probes, evaluation jobs and expected outcomes
by SHA-256. Do not regenerate the probes in an existing frozen checkout.
Model outputs are ignored under `benchmark_study/results/`.

## License

Original code and authored synthetic fixtures are MIT licensed. The four
ACI-Bench excerpts remain CC BY 4.0; see
[`benchmark_study/data/legacy/ATTRIBUTION.md`](benchmark_study/data/legacy/ATTRIBUTION.md).
Model and dependency licenses remain separate.
