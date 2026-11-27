# SPDX-License-Identifier: Apache-2.0
"""AMT1 RGB prediction, manifest inference, evaluation, and cache reproduction."""
import argparse
import json
from pathlib import Path

from .io import checked_relative, read_json, save_prediction, sha256, write_json


def build_parser() -> argparse.ArgumentParser:
    """Build public commands with explicit input, output and asset paths.

    Returns:
        Argument parser for the documented AMT1 entry points.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Predict one unlabeled RGB image.")
    predict.add_argument("--image", required=True, type=Path)
    predict.add_argument("--category", required=True)
    predict.add_argument("--prompt-name")
    predict.add_argument("--states", choices=("industrial", "clinical"), default="industrial")
    predict.add_argument("--geometry", choices=("ordinary", "classification-only", "ksdd-three-bands"), default="ordinary")
    predict.add_argument("--text-vocab", choices=("objsurf", "anatex"))
    predict.add_argument("--text-gate", type=float)
    predict.add_argument("--assets", type=Path, required=True, help="Relative-path model asset JSON.")
    predict.add_argument("--device", default="cuda:0")
    predict.add_argument("--bands", nargs=3, type=Path)
    predict.add_argument("--output", required=True, type=Path, help="New NPZ prediction path.")
    predict.add_argument("--observations", type=Path, help="Optional new AMT1-only observation NPZ.")
    infer = commands.add_parser("infer-manifest", help="Predict an ordered category manifest.")
    infer.add_argument("--manifest", type=Path, required=True)
    infer.add_argument("--data-root", type=Path, required=True, help="Root for manifest image/band paths.")
    infer.add_argument("--assets", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--device", default="cuda:0")
    infer.add_argument("--indices", nargs="*", type=int, help="Explicit subset for packaging validation.")
    replay = commands.add_parser("replay", help="Recompute all category predictions from fixed observations.")
    replay.add_argument("--manifests", type=Path, required=True)
    replay.add_argument("--artifacts", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--keys", nargs="*", help="Optional explicit dataset/category shard.")
    replay.add_argument("--shard-index", type=int, default=0)
    replay.add_argument("--shards", type=int, default=1)
    evaluate = commands.add_parser("evaluate", help="Compute eight applicable category metrics.")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    gt = evaluate.add_mutually_exclusive_group(required=True)
    gt.add_argument("--artifacts", type=Path, help="Explicit frozen GT artifact root.")
    gt.add_argument("--data-root", type=Path, help="Load masks directly from this data root.")
    evaluate.add_argument("--identity-map", type=Path, help="Explicit identity mapping for single-image predictions.")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default="cpu")
    check = commands.add_parser("check-data", help="Verify manifest paths and provided checksums.")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--data-root", type=Path, required=True)
    prepare = commands.add_parser("prepare-data", help="Create an admitted layout from recorded raw source paths.")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--source-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare_gt = commands.add_parser("prepare-gt", help="Read raw masks using the original 518-pixel convention.")
    prepare_gt.add_argument("--manifest", type=Path, required=True)
    prepare_gt.add_argument("--data-root", type=Path, required=True)
    prepare_gt.add_argument("--output", type=Path, required=True, help="New boolean GT.npy path.")
    aggregate = commands.add_parser("aggregate", help="Aggregate complete category results into dataset/domain means.")
    aggregate.add_argument("--manifests", type=Path, required=True)
    aggregate.add_argument("--metrics", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """Execute a public AMT1 entry point and print a compact JSON result.

    Raises:
        ValueError: Manifest identity, input paths, or reproduction checks fail.
        FileNotFoundError: Required data or assets are unavailable.
    """
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "predict":
        from .predictor import Predictor

        predictor = Predictor.from_config(args.assets, args.device)
        prediction = predictor.predict(
            args.image, args.category, prompt_name=args.prompt_name,
            states=args.states, text_vocab=args.text_vocab, text_gate=args.text_gate,
            geometry=args.geometry, band_images=args.bands,
            observation_output=args.observations)
        save_prediction(args.output, prediction)
        print(json.dumps(dict(output=str(args.output), image_score=prediction.image_score,
                              visual_forwards=predictor.forward_count)))
    elif args.command == "infer-manifest":
        from .predictor import Predictor

        manifest = read_json(args.manifest)
        selected = args.indices if args.indices is not None else range(len(manifest["samples"]))
        selected = list(selected)
        if len(selected) != len(set(selected)) or any(
                index < 0 or index >= len(manifest["samples"]) for index in selected):
            raise ValueError("Inference indices must be unique and within the manifest range.")
        manifest_hash = sha256(args.manifest)
        completed = {}
        for index in selected:
            path = args.output / f"{index:05d}.npz"
            marker_path = path.with_suffix(".complete.json")
            if not path.exists() and not marker_path.exists():
                continue
            expected = dict(sample_id=manifest["samples"][index]["id"], index=index,
                            manifest_sha256=manifest_hash)
            if not path.is_file() or not marker_path.is_file():
                raise ValueError(f"Partial prediction exists at index {index}. Use a new output directory.")
            marker = read_json(marker_path)
            if (marker.get("status") != "completed" or marker.get("identity") != expected
                    or marker.get("sha256") != sha256(path)):
                raise ValueError(f"Existing prediction identity or content differs at index {index}.")
            import numpy as np

            with np.load(path) as prediction:
                completed[index] = float(prediction["J"])
        pending = [index for index in selected if index not in completed]
        predictor = Predictor.from_config(args.assets, args.device) if pending else None
        results = []
        for index in selected:
            sample = manifest["samples"][index]
            if index in completed:
                results.append(dict(index=index, id=sample["id"], J=completed[index]))
                continue
            image = checked_relative(args.data_root, sample["image"])
            bands = None if not sample.get("bands") else [
                checked_relative(args.data_root, band["image"]) for band in sample["bands"]]
            # Only RGB and protocol metadata cross the predictor interface.
            prediction = predictor.predict(
                image, manifest["category"], prompt_name=manifest["prompt_name"],
                states=manifest["states"], text_vocab=manifest["text_vocab"],
                text_gate=manifest["text_gate"], geometry=manifest["geometry"], band_images=bands)
            identity = dict(sample_id=sample["id"], index=index, manifest_sha256=manifest_hash)
            save_prediction(args.output / f"{index:05d}.npz", prediction, identity=identity)
            results.append(dict(index=index, id=sample["id"], J=prediction.image_score))
        write_json(args.output / "inference.json", dict(samples=results,
                   visual_forwards=predictor.forward_count if predictor is not None else 0))
    elif args.command == "replay":
        from .replay import replay_category

        if args.shards < 1 or not 0 <= args.shard_index < args.shards:
            parser.error("Shard index must be in [0, shards).")
        manifests = sorted(args.manifests.glob("*.json"))
        selected = [path for path in manifests
                    if not args.keys or read_json(path)["key"] in args.keys]
        for path in selected[args.shard_index::args.shards]:
            report = replay_category(path, args.artifacts, args.output)
            print(json.dumps({name: report[name] for name in
                              ("status", "key", "count", "maximum_errors", "seconds")}), flush=True)
    elif args.command == "evaluate":
        from .evaluate import evaluate

        print(json.dumps(evaluate(args.manifest, args.predictions, args.artifacts,
                                  args.output, args.device, data_root=args.data_root,
                                  identity_map=args.identity_map)))
    elif args.command == "check-data":
        manifest = read_json(args.manifest)
        for sample in manifest["samples"]:
            image = checked_relative(args.data_root, sample["image"])
            if not image.is_file():
                raise FileNotFoundError(image)
            expected = sample.get("image_sha256")
            if expected is not None and sha256(image) != expected:
                raise ValueError(f"Image checksum differs for {sample['id']}.")
        print(json.dumps(dict(status="passed", count=len(manifest["samples"]))))
    elif args.command == "prepare-data":
        from .data import prepare_data

        print(json.dumps(prepare_data(args.manifest, args.source_root, args.output_root)))
    elif args.command == "prepare-gt":
        import numpy as np
        from .data import load_ground_truth

        masks = load_ground_truth(read_json(args.manifest), args.data_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            np.save(stream, masks, allow_pickle=False)
        print(json.dumps(dict(status="completed", shape=list(masks.shape), dtype="bool")))
    elif args.command == "aggregate":
        from .aggregate import aggregate

        result = aggregate(args.manifests, args.metrics, args.output)
        print(json.dumps(dict(status=result["status"], categories=result["count"])))
