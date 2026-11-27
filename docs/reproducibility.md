# Reproducibility record

## Scope and frozen authority

The archived full-benchmark package records the completed, audited AMT1 run. It covers 14 datasets, 66 categories and 20,277 original frames or images, with 21,075 two-resolution observation units. The exported benchmark uses the final authoritative AMT1 observations and predictions. Ordered manifests and point files bind the final sources through content hashes. The private archive retains all measured regressions against the same-source RADIOlogy baseline. The two current result tables use comparison values supplied by the project owner, whose source and protocol require separate verification.

The release task adds packaging checks and a bounded RGB verification. It does not rerun the full RGB benchmark or train a model. All numerical verification and archive generation ran on A40. Raw datasets are explicit external inputs.

## Fixed-observation replay

The public `replay` command completed all 66 categories in four independent category shards. Every process was waited for and exited successfully. Recomputed J, G, native and applicable Q have maximum absolute error 0.0 against the copied authoritative evidence. Exact counts are 20,277 J, 20,277 native fields and 16,824 localization maps. The other 3,453 images are classification-only. The slowest shard's elapsed wall time was approximately 925.5 seconds.

Replay supplies the historical official A map explicitly and recomputes the two-scale fusion, native readout, postprocessing and transport. It computes U, but a separate historical full-benchmark U asset was unavailable. U was independently checked against the original postprocessor on the seven RGB validation units. The published eight-metric point values are the audited existing results. Their source hashes and all 66 category values were verified. Full pixel sorting was not repeated during cache replay.

## Fresh RGB verification

The selection was frozen before inference using the canonical-first admitted identity within each required stratum. Seven evaluation units cover bottle normal and broken-large, carpet color, Woven anomaly, HeadCT classification, ISIC localization and one KSDD original frame with its three original bands. Eighteen visual forwards were performed.

The first unit ran through `python -m amt1 predict`. The remaining units used the same portable Predictor API. Source was relocated, the working directory was outside the project, and all loaded package/model module paths were recorded. No old development or runtime code was imported. Labels and GT were absent from the predictor interface.

Against historical evidence, G was exact and foreground patches were unchanged. Maximum D0 error was 2.2351741790771484e-8 and maximum J error was 1.862645149230957e-9. Maximum final Q error was 4.827976226806641e-6. Every selected localization unit exceeded the original 1e-7 historical Q tolerance. The classification-only unit passed its applicable tolerance. Fresh RGB inference therefore does not have historical bit identity or a demonstrated 1e-7 final-map guarantee.

On identical fresh inputs, the independently loaded original transport and postprocessor agree exactly with this release, including KSDD stitching. This localizes the observed discrepancy upstream of the final transport comparison. A unique cause of the small upstream differences was not established. GPU clustering may vary across executions, but that fact alone does not identify the cause of these observations. Seeds, arithmetic, precision and tolerance were preserved.

Measured own-job peak GPU allocation was 6,437,151,232 bytes, reservation 6,893,338,624 bytes, and host RSS 5,788,860,416 bytes. Ordinary inference uses two visual forwards and three-band KSDD uses six per original frame.

## Numerical, data and interface checks

The numerical script checks area geometry, source ties, constant and identity transport, native tail precision and KSDD frame assembly. The original W, transport and postprocessing were also compared independently on all seven fresh validation units with error 0.0.

Raw-mask loading matched the copied boolean GT exactly on all six localization units. Classification-only data rejected invented pixel GT. The two-image public evaluator accepted an explicit content-bound identity mapping and rejected a wrong mapping. Raw GT and frozen GT produced identical CPU metrics. DTD image/mask preparation and KSDD quality-95 three-band preparation matched the original data and band bytes.

Final public CLI checks cover help, all dataset/domain aggregate cells, duplicate/out-of-range indices, partial-output rejection, identity mismatch rejection and a completed-output resume with zero model forwards. The final aggregate values agree with exported tables to their displayed numeric precision. Both existing measured baseline and AMT1 are checked.

The two-image CPU/GPU metric comparison had maximum absolute error 5.53224e-13, within the original 1e-7 tolerance.

## Resume manifest inference

Repeat the same `infer-manifest` command with the same manifest and output directory
to resume a run. Each sample has an index-based NPZ filename and a completion marker.

```text
outputs/mvtec/bottle/
  00000.npz
  00000.complete.json
  ...
  inference.json
```

Before reusing a prediction, the CLI verifies its completion status, sample ID,
manifest index, manifest SHA256 and prediction SHA256. It computes pending samples
and reuses the verified scores. When all selected samples are complete, the run
records `visual_forwards` as `0` in `inference.json`.

A prediction missing its completion marker, or a marker missing its prediction,
raises a partial-output error. Changed sample identities or prediction contents
also raise an error. Use a fresh output directory for these cases and retain the
interrupted directory for diagnosis.

Use a separate output directory for each category or concurrent worker. The
single-image `predict` command requires a new destination filename. Manifest-based
resumption is handled by `infer-manifest`.

## Assets and source identity

The asset audit checked 265 benchmark/model files against their recorded SHA256 and sizes. All 126 official/final map source bindings were additionally checked against the original files, with different inodes. There are no symlinks under the release. Models, observations, predictions and GT are physically independent copies.

Exact package versions and model revisions are in `configs/environment.json` and `configs/assets_manifest.json`. RADIOlogy source revision is `7c36e32be4910f9b1bccf588ec14fb40f779f74e`. The actual C-RADIO source cache lacked an authenticated Git commit. Its recorded Python-tree fingerprint is `287d45ed3df2e80b6133f9c37555f604748be6e994cfb33e4fc4eb9e1f25c04d`. The fingerprint hashes sorted relative path, NUL, file SHA256 and newline records. Released source hashes and modifications are recorded in `third_party/provenance.json`.

## Clean source distribution

The source archive includes code, pinned configuration, result tables/manifests, documentation and third-party source/licenses. Model weights and benchmark artifacts have separate archives. Raw datasets remain external. Extract archives at a common project root so model paths in `configs/assets.json` remain valid. Asset archives use root-relative `artifacts/...` members.

The accompanying `SHA256SUMS`, public file manifest and asset manifests specify archive/file identities. `dist/clean_validation.json` records installation into an isolated target from the extracted source archive, CLI/import checks, the numerical script, a meaningful HeadCT cache replay and raw-GT evaluation. The clean check uses explicit external asset/data paths and performs zero additional visual forwards. Its report accompanies the immutable source archive and is excluded from ordinary source uploads.

## Benchmark caveats

Thirteen of fourteen reproduced baseline datasets align with the paper at its displayed precision. ClinicDB paper pixel AUROC/F1/AUPRO are 89.6/54.9/73.2 percent, while the measured baseline is 86.748363/49.089087/67.974964 percent. The cause is unconfirmed. KSDD uses the admitted 399-frame fallback. The medical classification admission uses the author's AdaCLIP test lists, and equivalence to the exact RADIOlogy paper manifest remains unverified. Task-inapplicable values remain NA. See `configs/dataset_sources.json` and the private archived evaluation protocol.
