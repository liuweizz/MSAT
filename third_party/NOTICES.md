# Third-party source and model terms

## RADIOlogy

Source revision `7c36e32be4910f9b1bccf588ec14fb40f779f74e` is included in `src/amt1/_vendor/radiology/` under Apache-2.0. The complete license is `RADIOlogy-LICENSE`. The necessary upstream scoring, foreground, preprocessing, backbone and metric routines remain identifiable.

The backbone loader uses explicit local model/source paths. The foreground and preprocessing modules contain the required functions extracted from their original modules. The exact pooled metric engine retains its validated function bodies. The public provenance file records original and released file hashes and all packaging changes.

Upstream paper source is [RADIOlogy](https://arxiv.org/abs/2610.40580). Consult the original project's citation metadata for authors and publication details.

## C-RADIO software

`C-RADIO/` includes the actual frozen source used by the validated environment, identified by its Python source-tree fingerprint. The local cache did not retain an authenticated upstream Git commit. The fingerprint and per-file hashes specify the source used here.

The included `C-RADIO/LICENSE` is the NVIDIA Source Code License for RADIO. Section 3.3 limits software use to non-commercial research or evaluation purposes. Original copyright and license notices remain in the source files.

The local-loading adaptation preserves the canonical `c-radio_v4-h` resource metadata while reading a configured checkpoint. The SigLIP2 adaptor uses the configured local snapshot and `local_files_only=True`. See `C-RADIO-MODIFICATIONS.md` and `provenance.json`.

## Model weights

C-RADIOv4-H weights have separate governing terms in the pinned model card and NVIDIA Open Model License Agreement. Copies are included as `C-RADIO-MODEL_CARD.md` and `C-RADIO-NVIDIA-OPEN-MODEL-LICENSE.pdf`. The model card identifies revision `0057b339059c0b9e1b4ba996f975410ebbfdfcc8`.

The pinned SigLIP2 model card specifies Apache-2.0. Its model card and license text are `SigLIP2-MODEL_CARD.md` and `SigLIP2-LICENSE`. Model weights remain in the ignored local asset directory and in a separately checksummed asset bundle.

## New contributions

The release includes AMT1-owned fusion, transport, inference interfaces, portable I/O and release documentation. The project owner selected Apache-2.0 for AMT1 new code and documentation. The complete text is the top-level `LICENSE`. Included upstream materials retain their own licenses.
