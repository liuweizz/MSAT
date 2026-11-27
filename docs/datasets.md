# Admitted data and ordered evaluation manifests

The private experiment archive contains one portable manifest per category, preserving ordered sample IDs, labels, prompt names, foreground vocabulary, task applicability and image paths relative to an explicitly supplied data root. The completed benchmark covers 66 categories and 20,277 evaluation images or frames. Supply external evaluation manifests to the manifest-based CLI commands.

For the 11 industrial/medical-localization datasets, point `--data-root` at a directory matching the exported `datasets` path group. The manifest records the actual prepared layout used for MVTec, VisA, BTAD, MPDD, KSDD, DAGM, DTD-Synthetic, ISIC, ClinicDB, ColonDB and TN3K. Supply masks at their listed relative paths for source-data preparation or use the checksummed prepared `GT.npy` artifacts for exact benchmark evaluation. The predictor receives RGB and prompt/geometry metadata.

Medical classification manifests for HeadCT, BrainMRI and Br35H use the original accessible AdaCLIP-author `meta.json` test lists. Their `--data-root` is the corresponding author's archive root, containing `headct/`, `brain_mri/` or `br35h/` as listed in each manifest. Counts are 200, 253 and 3,000 respectively. Pixel GT is absent for these datasets. The exact RADIOlogy paper test-file identity remains unverified.

KSDD uses all 399 admitted original frames as the recorded fallback, including 52 anomalous frames. Each frame is split into three bands using rounded original row boundaries. Original prepared JPEG bands use quality 95. A frame manifest contains each band's path and original crop edges. Evaluation remains at original-frame level, using separately resized/blurred bands and one whole-frame transport.

All paths and labels are frozen from the measured benchmark. There is no randomized resplitting. Raw data, masks and dataset rights must be obtained from the respective providers. The local artifact bundle supplies exact admitted masks and scores for historical replay, while RGB inputs remain explicitly configured.

Portable data preparation should retain these source identities. The `check-data` command validates every image path and all provided file hashes. `infer-manifest` accepts the same manifests, and `evaluate` consumes the output predictions in their original order. This supports already admitted layouts without a hidden dataset registry or machine-specific import.

`configs/data_preparation/` provides the same frozen samples with additional original
source-relative image and mask paths. `prepare-data` copies the original image bytes,
preserves grayscale mask values through Pillow `convert("L")`, and generates the
recorded KSDD JPEG bands. It preserves existing matching files and reports conflicting
outputs. Set `--source-root` to the recorded provider-tree layout and `--output-root`
to the root used by the evaluation manifests. Download URLs, source checksums and
recorded split/rights caveats are in `configs/dataset_sources.json`.

`evaluate --data-root` loads masks directly from this prepared public layout.
`prepare-gt` exports the same boolean arrays independently of predictions. Both use
the original-frame nearest-neighbor 518 geometry, including whole-frame KSDD masks.

## Single-image medical inference

Medical images use `--states clinical`. The following examples use the category,
prompt name and geometry recorded in the corresponding data-preparation manifests.
Replace the image path with a prepared RGB-readable input and choose a new output
filename for each prediction.

HeadCT uses classification-only geometry.

```bash
python -m amt1 predict --image /path/to/headct.png \
  --category headct --prompt-name "head CT" --states clinical \
  --geometry classification-only --assets configs/assets.json \
  --device cuda:0 --output outputs/headct.npz
```

The resulting NPZ contains `J`, `G` and `native`. For this geometry, the Python API
returns `None` for its localization maps.

TN3K uses ordinary localization geometry and a thyroid-ultrasound prompt.

```bash
python -m amt1 predict --image /path/to/thyroid.png \
  --category tn3k --prompt-name "thyroid ultrasound" --states clinical \
  --geometry ordinary --assets configs/assets.json \
  --device cuda:0 --output outputs/tn3k.npz
```

Clinical states select the `anatex` foreground vocabulary and a `0.014` gate
threshold by default. For benchmark evaluation, use `infer-manifest` with the exact
external manifest to carry the frozen prompt settings, sample order and identity
into the prediction completion markers.
