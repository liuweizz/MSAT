# Datasets and Evaluation

MSAT is evaluated across **14 benchmarks** (7 industrial and 7 medical datasets, 66 categories, 20,277 images/frames):

- **Industrial (Detection & Localization)**: MVTec-AD, VisA, BTAD, MPDD, KolektorSDD (KSDD), DAGM, DTD-Synthetic
- **Medical Detection**: HeadCT (200 images), BrainMRI (253 images), Br35H (3,000 images)
- **Medical Localization**: ISIC, CVC-ClinicDB, CVC-ColonDB, TN3K

Download links and source information are listed in `configs/dataset_sources.json`.

## Data Preparation

Preparation manifests are provided under `configs/data_preparation/`.

1. **Prepare datasets** into the standard evaluation layout (copies images, converts masks to grayscale, and splits extreme-aspect-ratio KSDD frames into 3 non-overlapping bands):

```bash
python -m amt1 prepare-data --source-root /path/to/raw_datasets --output-root /path/to/prepared_datasets
```

2. **Verify dataset files and checksums**:

```bash
python -m amt1 check-data --data-root /path/to/prepared_datasets
```

3. **Run manifest-based inference and evaluation**:
   - Use `python -m amt1 infer-manifest` with the category manifest and `--data-root`.
   - Use `python -m amt1 evaluate --data-root /path/to/prepared_datasets` to compute image-level (`AUROC`, `max-F1`, `AP`) and pixel-level (`AUROC`, `max-F1`, `AP`, `AUPRO@0.3`, `PRO@1%FPR`) metrics.
   - Use `python -m amt1 prepare-gt` to export ground-truth boolean masks resized to `518 × 518` (whole-frame masks for KSDD).

### Dataset-Specific Notes

- **Medical classification datasets (`HeadCT`, `BrainMRI`, `Br35H`)**: Follow the test splits from [AdaCLIP](https://github.com/caoyunkang/AdaCLIP) (`headct/`, `brain_mri/`, `br35h/`). Image-level metrics are reported without pixel ground truth.
- **KSDD**: Evaluated on all 399 full-frame images (52 anomalous). Each frame is split into 3 non-overlapping strips along its long axis for visual encoding and reassembled for whole-frame transport and evaluation.

## Single-Image Medical Inference

Medical datasets use clinical prompt states (`--states clinical`), which default to the `anatex` foreground vocabulary and a foreground-gating threshold of `0.014`.

### Medical Classification-Only (`HeadCT`, `BrainMRI`, `Br35H`)

```bash
python -m amt1 predict \
  --image /path/to/headct.png \
  --category headct \
  --prompt-name "head CT" \
  --states clinical \
  --geometry classification-only \
  --assets configs/assets.json \
  --device cuda:0 \
  --output outputs/headct.npz
```

The output `.npz` contains `J`, `G`, and `native` (localization maps are omitted under `classification-only`).

### Medical Localization (`ISIC`, `CVC-ClinicDB`, `CVC-ColonDB`, `TN3K`)

```bash
python -m amt1 predict \
  --image /path/to/thyroid.png \
  --category tn3k \
  --prompt-name "thyroid ultrasound" \
  --states clinical \
  --geometry ordinary \
  --assets configs/assets.json \
  --device cuda:0 \
  --output outputs/tn3k.npz
```
