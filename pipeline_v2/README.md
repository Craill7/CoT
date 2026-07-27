# CoT Pipeline V2.1

Versioned pipeline that reuses the project's current `select_v2` and Compress
V4.2 implementations without modifying their outputs.

V2.1 adds lexical-boundary atomic segmentation, a typed LLM answer gate for
the high-quality route, an answer audit pool, and one diagnosis-schema retry.

For the current V2.1.5 architecture, state machine, record schemas, code map,
and real smoke evidence, see [ARCHITECTURE.md](ARCHITECTURE.md).

## State machine

`normalize -> select -> global score -> cluster -> high/low`

- High-quality records pass the deterministic answer gate and are accepted.
- Low-quality records are analyzed by Compress V4.2, diagnosed with a separate
  block-to-block dependency graph, optionally repaired once with a structured
  patch, re-analyzed, and verified.
- Every record reaches `accepted` or `rejected`; there is no unlimited retry.

The V4.2 `dependency_graph` field remains block-to-target. Pipeline V2 adds
`block_dependencies`, `dependency_state`, and structured `gaps` in its diagnosis
records, preserving compatibility with current statistics and SFT builders.

## Entrypoint

Run from the CoT repository root:

```bash
python -m pipeline_v2.run \
  --input path/to/input.jsonl \
  --output-dir results/pipeline_v2/run_name \
  --base-url http://127.0.0.1:8000/v1 \
  --model /ky200t/models/Qwen/Qwen2.5-32B-Instruct \
  --select-mode auto \
  --clusterer score \
  --llm-max-tokens 4096 \
  --max-rewrite-rounds 1 \
  --resume
```

Use `--clusterer ifd_kmeans` for the existing IFD/embedding/KMeans route. This
loads the embedding model and therefore should only be used when its GPU memory
has been reserved. Framework tests use `--mock --clusterer score` and do not
call a model.

## Outputs

The output directory contains:

```text
normalized.jsonl
selected.jsonl
high_quality.jsonl
low_quality.jsonl
answer_audit.jsonl
diagnosed.jsonl
rewritten.jsonl
accepted.jsonl
rejected.jsonl
lineage.jsonl
stats.json
```

All JSONL stages use `sample_id + config_fingerprint` de-duplication. Re-running
the same configuration does not append a successful stage twice.

## Tests

```bash
python -m unittest discover -s pipeline_v2/tests -v
```

The suite covers selection routing, open/closed/cyclic dependency diagnosis,
structured patch scope, answer preservation, one-rewrite terminal behavior,
and restart idempotency. `fixtures/dependency_cases.json` contains compact,
sanitized snapshots of the actual `shard_3/sample_28` and
`shard_0/sample_46` V4.2 records; `fixtures/smoke.json` is the no-model
end-to-end fixture.
