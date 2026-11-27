# Model Setup

MSAT uses two frozen vision–language checkpoints:

| Model | Hugging Face Repository | Revision |
| --- | --- | --- |
| C-RADIOv4-H | [`nvidia/C-RADIOv4-H`](https://huggingface.co/nvidia/C-RADIOv4-H) | `0057b339059c0b9e1b4ba996f975410ebbfdfcc8` |
| SigLIP2 Giant | [`google/siglip2-giant-opt-patch16-384`](https://huggingface.co/google/siglip2-giant-opt-patch16-384) | `a713301b217d38485fb2204c808367d10bc3cc40` |

Exact filenames and SHA256 checksums are listed in `configs/assets_manifest.json`.

## Directory Layout

Download the C-RADIOv4-H checkpoint into `artifacts/models/radio/` and the complete SigLIP2 snapshot (weights, config, processor, and tokenizer files) into `artifacts/models/siglip2/`:

```text
MSAT/
├── configs/assets.json
├── third_party/C-RADIO/
└── artifacts/models/
    ├── radio/
    │   └── c-radio_v4-h_half.pth.tar
    └── siglip2/
```

## Configuration (`configs/assets.json`)

The default `configs/assets.json` uses paths relative to the config file directory (absolute paths are also supported):

| Key | Description |
| --- | --- |
| `radio_source` | Path to the C-RADIO source directory containing `hubconf.py` (`third_party/C-RADIO/`) |
| `checkpoint` | Path to the C-RADIOv4-H checkpoint (`c-radio_v4-h_half.pth.tar`) |
| `teacher_directory` | Path to the local SigLIP2 snapshot directory |

## Runtime Notes

- Weights are loaded in `float32` eval mode with `autocast` and `TF32` disabled for reproducible inference.
- Hugging Face / Transformers models are loaded offline from the configured local directories.
- Upstream model weights remain subject to their respective licenses (see `third_party/NOTICES.md`).
