"""Extracted RADIOlogy functions, Apache-2.0. See third_party notices."""
from typing import Callable
from torchvision import transforms as T

def default_transform(img_size: int, normalize: bool = False) -> Callable:
    """Resize to a square ``img_size`` and map to a ``[0, 1]`` tensor.

    ``normalize=False`` (the default) is what RADIO wants: it applies
    ImageNet mean/std internally via its input conditioner, so feeding it a
    plain ``[0, 1]`` tensor is correct.
    """
    steps = [
        T.Resize((img_size, img_size),
                 interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
    ]
    if normalize:
        steps.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return T.Compose(steps)

