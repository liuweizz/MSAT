# SPDX-License-Identifier: Apache-2.0
"""Label-free RGB inference with two-scale semantic and structural evidence."""
from io import BytesIO
import math
import os
from pathlib import Path
import threading

import numpy as np
from PIL import Image

from .core import Prediction, predict_observations
from .io import read_json

_MODEL_LOAD_LOCK = threading.Lock()


class Predictor:
    """Load explicit local assets and produce two-scale AMT1 predictions.

    Args:
        radio_source: Vendored C-RADIO directory with hubconf.py and license.
        checkpoint: C-RADIOv4-h checkpoint file.
        teacher_directory: Local SigLIP2 snapshot with processor and weights.
        device: PyTorch device such as cuda:0 or cpu.
        seed: Spherical-kmeans seed, fixed at zero for deterministic initialization.

    Raises:
        FileNotFoundError: A required explicit model asset is missing.
        ValueError: The requested seed differs from the configured method.
    """

    def __init__(self, radio_source: str | Path, checkpoint: str | Path,
                 teacher_directory: str | Path, device: str = "cuda:0", seed: int = 0):
        import torch
        from ._vendor.radiology import utils
        from ._vendor.radiology.backbones.radio import RadioBackbone

        if seed != 0:
            raise ValueError("The released AMT1 protocol specifies seed zero.")
        paths = [Path(radio_source).resolve(), Path(checkpoint).resolve(),
                 Path(teacher_directory).resolve()]
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Required local model asset is missing: {path}")
        self.device = torch.device(device)
        self.forward_count = 0
        self.banks = {}
        self.gate_banks = {}
        self.seed = seed
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        utils.set_seed(seed)
        # Serialize model loading because adaptor paths use process-wide variables.
        with _MODEL_LOAD_LOCK:
            os.environ.update(AMT1_RADIO_SOURCE=str(paths[0]),
                              AMT1_RADIO_CHECKPOINT=str(paths[1]),
                              AMT1_SIGLIP_DIR=str(paths[2]),
                              HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            self.backbone = RadioBackbone(version="c-radio_v4-h", device=device,
                                          img_size=448, want_text=True, want_sam=True)
        self.backbone.model.float().eval()
        self._prediction_lock = threading.Lock()

    @classmethod
    def from_config(cls, path: str | Path, device: str = "cuda:0"):
        """Resolve an asset configuration relative to the configuration file.

        Args:
            path: JSON file containing radio_source, checkpoint, and teacher_directory.
            device: Requested PyTorch device.

        Returns:
            Loaded Predictor using explicit local assets.
        """
        config_path = Path(path).resolve()
        config = read_json(config_path)
        return cls(*(config_path.parent / config[name] for name in
                     ("radio_source", "checkpoint", "teacher_directory")), device=device)

    def _anchors(self, category: str, name: str, states: str, vocabulary: str):
        from ._vendor.radiology import zeroshot

        key = (category, name, states)
        if key not in self.banks:
            normal, anomaly = zeroshot.STATE_VOCABULARIES[states]
            self.banks[key] = zeroshot.text_anchors(
                self.backbone, category, name=name,
                normal_states=normal, anomaly_states=anomaly)
        if vocabulary not in self.gate_banks:
            self.gate_banks[vocabulary] = zeroshot.fgbg_anchors(
                self.backbone, category, vocab=vocabulary)
        return self.banks[key], self.gate_banks[vocabulary]

    def _encode(self, image: Image.Image, anchors, gate_anchors, size: int):
        import torch
        from ._vendor.radiology import utils, zeroshot
        from ._vendor.radiology.preprocessing import default_transform

        self.backbone.set_img_size(size)
        tensor = default_transform(size, normalize=False)(image).unsqueeze(0)
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=False):
            features = self.backbone.encode(tensor)
            self.forward_count += 1
            patches = features.clip_text_patch[0]
            context = utils.neighborhood_aggregate(
                patches[None], features.grid, 5 if size == 448 else 9)[0]
            context_margin = zeroshot.patch_margin(context, *anchors)
            fields = []
            assignments = {}
            for regions in (8, 32):
                pooled, labels = zeroshot._region_pool(
                    patches, features.sam_patch[0], regions,
                    seed=self.seed, fg=None, kmeans_iters=15)
                region_margin = (pooled @ anchors[1] - pooled @ anchors[0])[labels]
                mixed = 0.5 * (region_margin + context_margin)
                fields.append(mixed)
                assignments[str(regions)] = labels.cpu().numpy()
            refined = [zeroshot.affinity_refine(
                field, features.patch_tokens[0], iters=1, temp=0.1, keep=0.5
            ) for field in fields]
            # Order region/context fields before affinity-refined fields in the reduction.
            margin = np.mean([field.cpu().numpy() for field in fields + refined], axis=0)
            global_score = float(zeroshot.patch_margin(features.clip_summary, *anchors)[0])
            foreground_anchor, background_anchor = gate_anchors
            gate = float(zeroshot.patch_margin(
                features.clip_summary, background_anchor, foreground_anchor)[0])
        return margin, global_score, gate, assignments

    def predict(self, image: str | Path | Image.Image, category: str,
                prompt_name: str | None = None, states: str = "industrial",
                text_vocab: str | None = None, text_gate: float | None = None,
                geometry: str = "ordinary", band_images: list[str | Path] | None = None,
                observation_output: str | Path | None = None) -> Prediction:
        """Predict an unlabeled image using explicit prompt and geometry metadata.

        Args:
            image: RGB-readable path or a Pillow image. Masks and labels are unused.
            category: Object or tissue category used in the prompt ensemble.
            prompt_name: Explicit object description, matching the benchmark manifest.
            states: industrial or clinical prompt state vocabulary.
            text_vocab: objsurf or anatex foreground vocabulary.
            text_gate: Foreground exemption threshold, .016 or .014 respectively.
            geometry: ordinary, classification-only, or ksdd-three-bands.
            band_images: Optional three admitted JPEG band paths for KSDD replay.
            observation_output: Optional new NPZ containing AMT1-only margins,
                foreground, global score and fresh region assignments.

        Returns:
            Prediction containing the image score and applicable localization map.

        Raises:
            ValueError: Geometry, prompt metadata or KSDD bands are incompatible.
        """
        from ._vendor.radiology import zeroshot

        if geometry not in ("ordinary", "classification-only", "ksdd-three-bands"):
            raise ValueError(f"Unsupported inference geometry {geometry!r}.")
        if states not in zeroshot.STATE_VOCABULARIES:
            raise ValueError(f"Unsupported prompt-state vocabulary {states!r}.")
        if band_images is not None and geometry != "ksdd-three-bands":
            raise ValueError("Explicit band inputs require KSDD three-band geometry.")
        source = Image.open(image).convert("RGB") if not isinstance(image, Image.Image) else image.convert("RGB")
        vocabulary = text_vocab or ("anatex" if states == "clinical" else "objsurf")
        gate_threshold = text_gate if text_gate is not None else (.014 if states == "clinical" else .016)
        name = prompt_name or zeroshot.prompt_name(category)
        images = [source]
        if geometry == "ksdd-three-bands":
            if band_images:
                if len(band_images) != 3:
                    raise ValueError("KSDD requires exactly three ordered band images.")
                images = [Image.open(path).convert("RGB") for path in band_images]
            else:
                count = math.ceil(source.height / source.width) if source.height / source.width > 2 else 1
                if count != 3:
                    raise ValueError("The admitted KSDD geometry requires three bands.")
                images = []
                for index in range(3):
                    crop = source.crop((0, round(index * source.height / 3), source.width,
                                        round((index + 1) * source.height / 3)))
                    encoded = BytesIO()
                    crop.save(encoded, format="JPEG", quality=95)
                    encoded.seek(0)
                    images.append(Image.open(encoded).convert("RGB"))
        low, high, foreground, globals_ = [], [], [], []
        assignments = {}
        with self._prediction_lock:
            anchors, gate_anchors = self._anchors(category, name, states, vocabulary)
            for band, rgb in enumerate(images):
                observations = {}
                for size in (448, 672):
                    margin, global_score, gate, region_labels = self._encode(
                        rgb, anchors, gate_anchors, size)
                    observations[size] = margin
                    for regions, labels in region_labels.items():
                        assignments[f"band{band}_{size}_regions{regions}"] = labels
                    if size == 448:
                        if geometry == "ksdd-three-bands":
                            mask = np.ones(784, bool)
                        else:
                            # Adapt the in-memory RGB image to the foreground
                            # estimator's lossless PNG stream interface.
                            from .preprocessing import colour_foreground
                            mask = colour_foreground(rgb) | (gate < gate_threshold)
                        foreground.append(mask)
                        globals_.append(global_score)
                low.append(observations[448])
                high.append(observations[672])
        low_array, high_array = np.stack(low), np.stack(high)
        foreground_array = np.stack(foreground)
        global_array = np.asarray(globals_, np.float64)
        if observation_output is not None:
            destination = Path(observation_output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                np.savez(stream, **{"448_D0": low_array, "672_D0": high_array,
                                    "F": foreground_array, "G": global_array, **assignments})
        return predict_observations(low_array, high_array, foreground_array,
                                    global_array, geometry=geometry)
