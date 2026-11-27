# Explicit local model assets

The verified assets are listed with file-level SHA256 in `configs/assets_manifest.json`.

| Asset | Official repository | Revision |
| --- | --- | --- |
| C-RADIOv4-H | nvidia/C-RADIOv4-H | 0057b339059c0b9e1b4ba996f975410ebbfdfcc8 |
| SigLIP2 giant teacher | google/siglip2-giant-opt-patch16-384 | a713301b217d38485fb2204c808367d10bc3cc40 |

Download the exact filenames from the per-file official URLs in that manifest. Place the C-RADIO checkpoint in `artifacts/models/radio/` and the complete SigLIP2 snapshot in `artifacts/models/siglip2/`. The latter includes both safetensor shards, the index, configuration, processor and tokenizer files. Check each file against its SHA256 before running.

`third_party/C-RADIO/` contains the frozen source used by the measured environment. Its public source-tree checksums and upstream provenance are recorded in `third_party/provenance.json`. The SigLIP2 adaptor reads the explicitly configured local directory. The RADIOlogy backbone loads the explicit local C-RADIO checkpoint while preserving the canonical version's resource metadata.

The checkpoint's stored half-precision weights are loaded into the validated float32/eval runtime. Teacher heads and the text adaptor retain the original architecture and tokenizer. Runtime visual inference disables autocast and TF32. Model construction uses the supplied local source and weights.

The local release includes physical copies of these assets under the ignored `artifacts/` directory. GitHub source uploads exclude the weights. Any separate distribution of weights remains governed by the upstream model licenses and model cards.

## Configure asset paths

The supplied `configs/assets.json` expects this project-relative layout.

```text
MSAT/
  configs/assets.json
  third_party/C-RADIO/
  artifacts/models/
    radio/c-radio_v4-h_half.pth.tar
    siglip2/
```

| Configuration key | Expected content |
| --- | --- |
| `radio_source` | C-RADIO source directory containing `hubconf.py` |
| `checkpoint` | C-RADIOv4-H checkpoint file |
| `teacher_directory` | Complete local SigLIP2 snapshot directory |

`Predictor.from_config` resolves relative values against the directory containing
the JSON file. Moving the JSON file requires updating its relative values to match
the new location. Absolute asset paths can also be supplied in the JSON.

Inference uses offline Hugging Face and Transformers loading. Prepare both model
snapshots before constructing the predictor. Missing asset paths raise
`FileNotFoundError` with the resolved location. Compare downloaded files with
`configs/assets_manifest.json` to check exact filenames, byte sizes and SHA256.
