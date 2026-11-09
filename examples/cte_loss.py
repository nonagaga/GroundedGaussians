import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Original hard (non-differentiable) detector -- use for eval / label checks.
# ---------------------------------------------------------------------------
def detect_track_centroid(frame, band_frac, hsv_lower, hsv_upper, min_frac=0.002):
    """
    Find the x centroid of orange pixels in the bottom `band_frac` of the frame.
    Returns x (float, full-frame pixel coords) or None if no track is found.
    `frame` is a BGR uint8 image.
    """
    h, _ = frame.shape[:2]
    band_h = max(1, int(round(h * band_frac)))
    band = frame[h - band_h :]

    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] < min_frac * mask.size:
        return None
    return m["m10"] / m["m00"]


def hard_cte(image, band_frac, hsv_lower, hsv_upper, min_frac=0.002):
    """image: [H, W, 3] RGB float tensor in [0, 1]. Returns CTE px, or None on failure."""
    rgb = (image.detach().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    x = detect_track_centroid(bgr, band_frac, hsv_lower, hsv_upper, min_frac)
    return None if x is None else x - image.shape[1] / 2.0


# ---------------------------------------------------------------------------
# Differentiable torch port -- use for the training loss.
# ---------------------------------------------------------------------------
def rgb_to_hsv_cv(rgb: torch.Tensor) -> torch.Tensor:
    """[..., 3] RGB in [0,1] -> HSV in OpenCV uint8 scale (H 0-180, S/V 0-255)."""
    r, g, b = rgb.unbind(-1)
    v, _ = rgb.max(-1)
    mn, _ = rgb.min(-1)
    c = v - mn
    c_safe = c.clamp_min(1e-6)
    s = c / v.clamp_min(1e-6)

    h = torch.where(
        v == r,
        ((g - b) / c_safe) % 6.0,
        torch.where(v == g, (b - r) / c_safe + 2.0, (r - g) / c_safe + 4.0),
    )
    h = torch.where(c > 1e-6, h * 30.0, torch.zeros_like(h))  # 60 deg / 2 = 30
    return torch.stack([h, s * 255.0, v * 255.0], dim=-1)


def soft_track_centroid(
    images, band_frac, hsv_lower, hsv_upper, min_frac=0.002, tau=(2.0, 10.0, 10.0)
):
    """
    images: [B, H, W, 3] RGB in [0, 1].
    Returns (centroid_x [B], found [B] bool, mass [B], min_mass float).
    """
    B, H, W, _ = images.shape
    band_h = max(1, int(round(H * band_frac)))
    band = images[:, H - band_h :].clamp(0, 1)

    hsv = rgb_to_hsv_cv(band)
    lo = torch.tensor(hsv_lower, dtype=hsv.dtype, device=hsv.device)
    hi = torch.tensor(hsv_upper, dtype=hsv.dtype, device=hsv.device)
    t = torch.tensor(tau, dtype=hsv.dtype, device=hsv.device)

    # Soft cv2.inRange: product of per-channel soft step functions.
    m = (torch.sigmoid((hsv - lo) / t) * torch.sigmoid((hi - hsv) / t)).prod(-1)
    m = m[:, None]  # [B, 1, h, W]

    # Soft MORPH_OPEN with a 3x3 kernel: erode then dilate.
    m = -F.max_pool2d(-m, 3, 1, 1)
    m = F.max_pool2d(m, 3, 1, 1)

    mass = m.sum(dim=(1, 2, 3))
    min_mass = min_frac * band_h * W
    col = m.sum(dim=(1, 2))  # [B, W]
    xs = torch.arange(W, device=m.device, dtype=m.dtype)
    centroid = (col * xs).sum(-1) / mass.clamp_min(1e-6)
    return centroid, mass >= min_mass, mass, min_mass


def cte_loss(
    images,
    cte_labels,
    band_frac,
    hsv_lower,
    hsv_upper,
    min_frac=0.002,
    tau=(2.0, 10.0, 10.0),
    label_width=None,
    hinge_weight=1.0,
):
    """
    L1 between detected CTE and labeled CTE, normalized by image width so that
    the worst case (detection failure) is exactly 1.0.

    images:     [B, H, W, 3] RGB render in [0, 1]
    cte_labels: [B] pixels (NaN = unlabeled, skipped)
    label_width: width (px) the labels were measured at; labels are rescaled
                 to the render width if it differs.
    """
    B, H, W, _ = images.shape
    cte_labels = cte_labels.to(images.dtype)
    if label_width is not None:
        cte_labels = cte_labels * (W / label_width)

    cx, found, mass, min_mass = soft_track_centroid(
        images, band_frac, hsv_lower, hsv_upper, min_frac, tau
    )
    pred = cx - W / 2.0
    err = (pred - cte_labels).abs().clamp(max=W)
    # Detection failed -> max CTE error (= image width).
    err = torch.where(found, err, torch.full_like(err, float(W)))
    loss = err / W

    # A failed detection has a constant error and therefore no gradient.
    # This hinge pushes the splats to produce more track-colored pixels.
    loss = loss + hinge_weight * F.relu(1.0 - mass / min_mass) * (~found).float()

    valid = ~torch.isnan(cte_labels)
    if not valid.any():
        return images.sum() * 0.0
    return loss[valid].mean()