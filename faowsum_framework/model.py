"""Shared mathematical primitives for the final FA-OWSum model."""

from dataclasses import dataclass

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class ModelConfig:
    """Fixed settings used by the retained prediction pipeline."""

    context_m: int = 60#预先缓存的候选数；
    candidate_m: int = 30#真正参与标签重排的候选数；
    top_k: int = 10#最终邻居数；
    smoothing: float = 1.0#对数优势平滑；
    feature_weight_cap: float = 8.0#MARTS对数优势上限；
    eta_shrink: float = 5.0#η正则强度；
    eta_grid_size: int = 21#η搜索点数；
    inner_splits: int = 5#内层交叉拟合折数；
    fingerprint_size: int = 2048
    fingerprint_radius: int = 2


DEFAULT_CONFIG = ModelConfig()


def ecfp_fingerprints(molecules, *, radius=2, fp_size=2048):#生成ECFP二值指纹。
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=fp_size,
        includeChirality=True,
    )
    return [generator.GetFingerprint(molecule) for molecule in molecules]


class TanimotoSimilarityCache:#预先保存所有分子两两之间的Tanimoto相似度，避免每个交叉验证折重新计算指纹相似度。

    def __init__(self, fingerprints):
        blocks = [
            np.asarray(
                DataStructs.BulkTanimotoSimilarity(
                    fingerprints[index], fingerprints[:index]
                ),
                dtype=np.float64,
            )
            for index in range(1, len(fingerprints))
        ]
        self.values = np.concatenate(blocks) if blocks else np.empty(0)

    def take(self, rows, columns):
        rows = np.asarray(rows, dtype=np.int64).reshape(-1, 1)
        columns = np.asarray(columns, dtype=np.int64).reshape(1, -1)
        high = np.maximum(rows, columns)
        low = np.minimum(rows, columns)
        indices = high * (high - 1) // 2 + low
        result = np.ones(np.broadcast_shapes(rows.shape, columns.shape))
        off_diagonal = rows != columns
        result[off_diagonal] = self.values[indices[off_diagonal]]
        return result


def _binary_csr(values):#把稠密或稀疏输入统一转换为二值CSR稀疏矩阵：
    if sparse.issparse(values):
        result = values.tocsr(copy=True)
        result.data = (result.data > 0).astype(np.float64)
        result.eliminate_zeros()
        return result
    return sparse.csr_matrix(np.asarray(values) > 0, dtype=np.float64)


def normalize_rows(values):#对每一行进行归一化

    values = np.asarray(values, dtype=np.float64)
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(
        values,
        totals,
        out=np.zeros_like(values),
        where=totals > 0,
    )


def distribution_rows(values):#防止全零分子导致总概率为0。
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(
        values,
        totals,
        out=np.full_like(values, 1.0 / values.shape[1]),
        where=totals > 0,
    )


def normalized_targets(y):
    return normalize_rows(np.asarray(y, dtype=np.float64))


def fit_weights(x, y):#计算FA-OWSum的全局forward和reverse权重。
    x_binary = _binary_csr(x)
    y_binary = _binary_csr(y)
    joint = (x_binary.T @ y_binary).toarray()
    feature_counts = np.asarray(x_binary.sum(axis=0)).ravel()
    label_counts = np.asarray(y_binary.sum(axis=0)).ravel()
    n_labels = y_binary.shape[1]

    forward_pr = np.divide(
        joint,
        label_counts,
        out=np.zeros_like(joint),
        where=label_counts > 0,
    )
    reverse_pr = np.divide(
        joint,
        feature_counts[:, None],
        out=np.zeros_like(joint),
        where=feature_counts[:, None] > 0,
    )
    feature_totals_per_label = joint.sum(axis=0)
    term_frequency = np.divide(
        joint,
        feature_totals_per_label,
        out=np.zeros_like(joint),
        where=feature_totals_per_label > 0,
    )

    label_support = np.count_nonzero(joint > 0, axis=1)
    idf = np.zeros_like(joint)
    ordinary = (label_support > 0) & (label_support < n_labels)
    idf[ordinary] = np.log10(n_labels / label_support[ordinary])[:, None]
    return forward_pr * idf, reverse_pr * term_frequency


def learn_alpha(forward, reverse, target):#用闭式最小二乘解学习forward与reverse的融合系数α。

    direction = np.asarray(reverse) - np.asarray(forward)
    residual = np.asarray(target) - np.asarray(forward)
    denominator = float(np.sum(direction * direction))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    value = float(np.sum(direction * residual) / denominator)
    return float(np.clip(value, 0.0, 1.0))


def learn_alpha_crossentropy(forward, reverse, target):
    """在训练OOF分数上用一维有界交叉熵优化融合系数α。"""

    forward = np.asarray(forward, dtype=np.float64)
    reverse = np.asarray(reverse, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    def objective(alpha):
        prediction = distribution_rows((1.0 - alpha) * forward + alpha * reverse)
        return float(
            -np.mean(
                np.sum(
                    target * np.log(np.clip(prediction, 1e-12, 1.0)),
                    axis=1,
                )
            )
        )

    result = minimize_scalar(
        objective,
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 1000},
    )
    # 有界求解器通常返回内部点；显式比较端点以覆盖α恰为0或1的情况。
    return min(
        (objective(0.0), 0.0),
        (objective(1.0), 1.0),
        (float(result.fun), float(result.x)),
    )[1]


def feature_weights(x_train, y_train, config):#计算标签特异SMARTS权重

    feature_count = np.asarray(x_train.sum(axis=0)).ravel()[:, None]
    positive_count = np.asarray(x_train.T @ y_train, dtype=float)
    positive_n = y_train.sum(axis=0)[None, :]
    negative_count = feature_count - positive_count
    negative_n = len(y_train) - positive_n
    smooth = config.smoothing
    positive_logit = np.log(
        (positive_count + smooth) / (positive_n - positive_count + smooth)
    )
    negative_logit = np.log(
        (negative_count + smooth) / (negative_n - negative_count + smooth)
    )
    result = np.clip(
        positive_logit - negative_logit, 0.0, config.feature_weight_cap
    )
    result[feature_count[:, 0] == 0] = 0.0
    return result


__all__ = [
    "DEFAULT_CONFIG",
    "ModelConfig",
    "TanimotoSimilarityCache",
    "distribution_rows",
    "ecfp_fingerprints",
    "feature_weights",
    "fit_weights",
    "learn_alpha",
    "learn_alpha_crossentropy",
    "normalize_rows",
    "normalized_targets",
]
