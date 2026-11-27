"""Extracted RADIOlogy functions, Apache-2.0. See third_party notices."""
import numpy as np

def _colour_fg(path: str, grid, border: int = 12, border_frac: float = 0.15,
               dilate: int = 1, probe: int = 0, probe_frac: float = 0.5,
               border_scale: float = None):
    """Patch-grid foreground mask from the distance to the median border colour.

    Returns ``(mask, d)``: the ``(N,)`` bool mask and Cohen's d between border
    and interior of the distance field. Model-free, but blind to defects that
    show background through the object (e.g. holes).
    """
    from PIL import Image
    gh, gw = grid
    im = np.asarray(Image.open(path).convert("RGB").resize((gw * 8, gh * 8)), np.float32)
    # `border` sets the reference-colour band (fixed pixel count unless
    # border_scale is given); `border_frac` below is only for the effect size.
    b = border if border_scale is None else max(1, int(round(border_scale * min(im.shape[:2]))))
    fr = np.concatenate([im[:b].reshape(-1, 3), im[-b:].reshape(-1, 3),
                         im[:, :b].reshape(-1, 3), im[:, -b:].reshape(-1, 3)])
    d = np.linalg.norm(im - np.median(fr, 0), axis=-1)
    h, e = np.histogram(d, bins=128)
    pr = h / h.sum(); c = 0.5 * (e[1:] + e[:-1])
    w0 = np.cumsum(pr); w1 = 1 - w0
    m0 = np.cumsum(pr * c) / np.maximum(w0, 1e-12)
    m1 = ((pr * c).sum() - np.cumsum(pr * c)) / np.maximum(w1, 1e-12)
    t = c[np.nanargmax(w0 * w1 * (m0 - m1) ** 2)]
    fg = base = (d > t).reshape(gh, 8, gw, 8).max(axis=(1, 3))
    if probe > 1:
        # If the dilated mask covers most of the image, skip masking it.
        from scipy.ndimage import maximum_filter
        if maximum_filter(base, size=probe).mean() >= probe_frac:
            return np.ones(gh * gw, bool), 0.0
    if dilate > 1:
        # Widen the mask's pooling block from 8px to 8*dilate px, aligned to
        # the patch grid.
        from scipy.ndimage import maximum_filter
        fg = maximum_filter(fg, size=dilate)

    # Border/interior effect size on the pixel-resolution distance field.
    bh = max(1, int(round(border_frac * min(d.shape))))
    is_border = np.zeros(d.shape, bool)
    is_border[:bh] = is_border[-bh:] = True
    is_border[:, :bh] = is_border[:, -bh:] = True
    db, di = d[is_border], d[~is_border]
    pooled = np.sqrt(0.5 * (db.var() + di.var()))
    eff = (db.mean() - di.mean()) / max(float(pooled), 1e-6)
    return fg.reshape(-1), float(eff)

