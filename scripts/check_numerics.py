# SPDX-License-Identifier: Apache-2.0
"""Check precision-sensitive fusion, scoring, and transport operations."""
import json

import numpy as np

from amt1.core import (area_weights, fuse, monotone_transport, native_tail,
                       postprocess, stitch_bands)


def main() -> None:
    """Check analytic fixtures and report maximum observed interface errors.

    Raises:
        AssertionError: A numerical operation violates its specified invariant.
    """
    weights = area_weights()
    assert weights.shape == (28, 42)
    assert np.max(np.abs(weights.sum(axis=1) - 1)) <= 1e-12
    high = np.arange(1764, dtype=np.float32) / np.float32(2048)
    low = np.arange(784, dtype=np.float32) / np.float32(1024)
    foreground = np.ones(784, bool)
    foreground[::3] = False
    native = fuse(low, high, foreground)
    assert native.dtype == np.float32
    assert np.array_equal(native[~foreground], np.full((~foreground).sum(), low.min()))
    promoted = np.sort(native.astype(np.float64))[-8:].mean(dtype=np.float64)
    assert native_tail(native) == promoted
    target = np.array([0., .25, .5, 1.], np.float32)
    tied = np.array([0., 0., 1., 1.], np.float32)
    anchored = monotone_transport(target, tied)
    assert np.array_equal(anchored, np.array([.125, .125, .75, .75], np.float32))
    constant = monotone_transport(target, np.zeros(4, np.float32))
    assert np.all(constant == np.float32(target.astype(np.float64).mean()))
    signed_zero = np.array([0., -0., .5, 1.], np.float32)
    assert monotone_transport(signed_zero, signed_zero).tobytes() == signed_zero.tobytes()
    band_native = np.stack([np.full(784, x, np.float32) for x in (.125, .25, .5)])
    stitched = stitch_bands(band_native)
    assert stitched.shape == (518, 518)
    assert np.all(stitched[:173] == .125)
    assert np.all(stitched[173:345] == .25)
    assert np.all(stitched[345:] == .5)
    assert postprocess(native).shape == (518, 518)
    print(json.dumps(dict(status="passed", row_weight_error=float(
        np.max(np.abs(weights.sum(axis=1) - 1))), checks=[
            "area geometry", "foreground floor", "promoted native tail",
            "source ties", "constant source", "signed-zero identity",
            "KSDD band boundaries and independent band blur"])))


if __name__ == "__main__":
    main()
