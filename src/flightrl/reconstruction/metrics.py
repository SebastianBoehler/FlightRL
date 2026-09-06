"""Evaluator-only alignment and geometric error; never called by the mapper."""

import numpy as np
from scipy.spatial import cKDTree


def alignment(source, target, scale=False):
    a = source - source.mean(0)
    b = target - target.mean(0)
    u, s, vt = np.linalg.svd(b.T @ a / len(a))
    d = np.eye(3)
    d[-1, -1] = np.linalg.det(u @ vt)
    r = u @ d @ vt
    if scale and (np.sum(a * a) < 1e-12 or np.sum(b * b) < 1e-12):
        raise ValueError("Degenerate trajectory cannot determine monocular scale")
    factor = float((s * np.diag(d)).sum() / (a * a).sum() * len(a)) if scale else 1.0
    t = target.mean(0) - factor * r @ source.mean(0)
    return factor, r, t


def score(estimated, truth, cloud, reference, mode):
    indices = [i for i, p in enumerate(estimated) if p is not None]
    result = {
        "tracked_frames": len(indices),
        "total_frames": len(estimated),
        "tracking_fraction": len(indices) / len(estimated) if len(estimated) else 0.0,
    }
    poses = np.array([estimated[i][:3, 3] for i in indices])
    degenerate = (
        mode == "rgb"
        and len(indices) >= 3
        and (
            np.sum((poses - poses.mean(0)) ** 2) < 1e-12
            or np.var(truth[indices, :3, 3], axis=0).sum() * len(indices) < 1e-12
        )
    )
    if len(indices) < 3 or degenerate:
        return {
            **result,
            "ate_rmse_m": None,
            "surface_accuracy_m": None,
            "surface_coverage": None,
            "unavailable_reason": "degenerate monocular scale"
            if degenerate
            else "fewer than three tracked frames",
        }
    gt = truth[indices, :3, 3]
    factor, r, t = (
        alignment(poses, gt, True) if mode == "rgb" else (1.0, np.eye(3), np.zeros(3))
    )
    aligned = factor * poses @ r.T + t
    error = np.linalg.norm(aligned - gt, axis=1)
    result.update(
        ate_rmse_m=float(np.sqrt(np.mean(error**2))),
        endpoint_error_m=float(error[-1]),
        evaluation_scale=factor,
        alignment="Sim(3), evaluator only"
        if mode == "rgb"
        else "initial camera frame only",
    )
    adjacent = [
        (j - 1, j) for j in range(1, len(indices)) if indices[j] == indices[j - 1] + 1
    ]
    result["relative_translation_rmse_m"] = (
        float(
            np.sqrt(
                np.mean(
                    [
                        np.sum(((aligned[b] - aligned[a]) - (gt[b] - gt[a])) ** 2)
                        for a, b in adjacent
                    ]
                )
            )
        )
        if adjacent
        else None
    )
    if len(cloud) and len(reference):
        # Equal weight per surface voxel, not per repeated camera sample.
        _, unique = np.unique(
            np.floor(reference / 0.08).astype(np.int64), axis=0, return_index=True
        )
        reference = reference[unique]
        reconstructed = factor * np.array([v[0] for v in cloud]) @ r.T + t
        distances = cKDTree(reference).query(reconstructed)[0]
        coverage = cKDTree(reconstructed).query(reference)[0]
        result.update(
            surface_accuracy_m=float(distances.mean()),
            surface_error_p95_m=float(np.quantile(distances, 0.95)),
            surface_coverage=float((coverage < 0.15).mean()),
        )
    else:
        result.update(surface_accuracy_m=None, surface_coverage=0.0)
    return result
