from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import json

import numpy as np


TrajClusterArtifact = Dict[str, Any]


def _to_2d_float32(array: np.ndarray) -> np.ndarray:
    casted = np.asarray(array, dtype=np.float32)
    if casted.ndim == 1:
        casted = casted.reshape(1, -1)
    return casted


def standardize_cluster_features(
    features: np.ndarray,
    feature_mean: Optional[np.ndarray] = None,
    feature_std: Optional[np.ndarray] = None,
) -> np.ndarray:
    feats = np.asarray(features, dtype=np.float32)
    if feats.size == 0:
        return feats.reshape(feats.shape[0], -1) if feats.ndim > 1 else np.zeros((0, 0), dtype=np.float32)

    mean = _to_2d_float32(feature_mean if feature_mean is not None else feats.mean(axis=0, keepdims=True))
    std = _to_2d_float32(feature_std if feature_std is not None else feats.std(axis=0, keepdims=True))
    std = np.where(std < 1e-6, 1.0, std)
    return (feats - mean) / std


def assign_cluster_labels(
    standardized_features: np.ndarray,
    cluster_centers: np.ndarray,
    batch_size: int = 4096,
) -> np.ndarray:
    feats = _to_2d_float32(standardized_features)
    centers = _to_2d_float32(cluster_centers)
    if feats.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if centers.shape[0] == 0:
        raise ValueError("cluster_centers must not be empty")

    labels = np.empty((feats.shape[0],), dtype=np.int64)
    for start in range(0, feats.shape[0], max(int(batch_size), 1)):
        end = min(start + max(int(batch_size), 1), feats.shape[0])
        chunk = feats[start:end]
        distances = np.sum((chunk[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels[start:end] = np.argmin(distances, axis=1).astype(np.int64)
    return labels


def save_traj_cluster_artifact(
    artifact_path: str,
    *,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    cluster_centers: np.ndarray,
    cluster_counts: Optional[np.ndarray] = None,
    tokens: Optional[np.ndarray] = None,
    token_labels: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    target_path = Path(artifact_path).expanduser()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "feature_mean": _to_2d_float32(feature_mean),
        "feature_std": _to_2d_float32(feature_std),
        "cluster_centers": _to_2d_float32(cluster_centers),
        "metadata_json": np.asarray(json.dumps(metadata or {}, sort_keys=True)),
    }
    if cluster_counts is not None:
        payload["cluster_counts"] = np.asarray(cluster_counts, dtype=np.int64)
    if tokens is not None:
        payload["tokens"] = np.asarray(tokens)
    if token_labels is not None:
        payload["token_labels"] = np.asarray(token_labels, dtype=np.int64)
    np.savez_compressed(target_path, **payload)
    return str(target_path)


def load_traj_cluster_artifact(artifact_path: str) -> TrajClusterArtifact:
    source_path = Path(artifact_path).expanduser()
    with np.load(source_path, allow_pickle=True) as data:
        metadata_raw = data["metadata_json"] if "metadata_json" in data else np.asarray("{}")
        if isinstance(metadata_raw, np.ndarray):
            metadata_text = metadata_raw.item() if metadata_raw.ndim == 0 else str(metadata_raw.tolist())
        else:
            metadata_text = str(metadata_raw)
        metadata = json.loads(metadata_text)

        artifact: TrajClusterArtifact = {
            "path": str(source_path),
            "feature_mean": _to_2d_float32(data["feature_mean"]),
            "feature_std": _to_2d_float32(data["feature_std"]),
            "cluster_centers": _to_2d_float32(data["cluster_centers"]),
            "metadata": metadata,
        }
        if "cluster_counts" in data:
            artifact["cluster_counts"] = np.asarray(data["cluster_counts"], dtype=np.int64)
        if "tokens" in data and "token_labels" in data:
            tokens = [str(token) for token in np.asarray(data["tokens"]).tolist()]
            token_labels = np.asarray(data["token_labels"], dtype=np.int64).tolist()
            artifact["token_to_cluster"] = {token: int(label) for token, label in zip(tokens, token_labels)}
        else:
            artifact["token_to_cluster"] = {}

    artifact["cluster_num_groups"] = int(metadata.get("cluster_num_groups", artifact["cluster_centers"].shape[0]))
    return artifact