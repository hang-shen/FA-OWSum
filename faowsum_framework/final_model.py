from __future__ import annotations
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import KFold
from .experiments import learn_label_thresholds
from .model import (
    DEFAULT_CONFIG,
    TanimotoSimilarityCache,
    distribution_rows,
    ecfp_fingerprints,
    feature_weights,
    fit_weights,
    learn_alpha,
    learn_alpha_crossentropy,
    normalize_rows,
    normalized_targets,
)


CONFIG = DEFAULT_CONFIG
ETA_VALUES = np.linspace(0.0, 1.0, CONFIG.eta_grid_size)
KAPPA_VALUES = (0.25, 0.5, 1.0, 2.0, 4.0, 10.0)


def make_fingerprints(molecules):#同时生成两种指纹
    bit = ecfp_fingerprints(
        molecules,
        radius=2,
        fp_size=CONFIG.fingerprint_size,
    )
    count_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=CONFIG.fingerprint_size,
        includeChirality=True,
    )
    count = [count_generator.GetCountFingerprint(molecule) for molecule in molecules]
    return {"ecfp4": bit, "count": count}


def make_similarity_caches(fingerprints):#分别为bit指纹和count指纹建立Tanimoto缓存
    return {
        name: TanimotoSimilarityCache(values)
        for name, values in fingerprints.items()
    }


def _similarities(caches, rows, columns):#提取指定查询分子与训练分子的
    ecfp4 = caches["ecfp4"].take(rows, columns)
    count = caches["count"].take(rows, columns)
    return ecfp4, 0.5 * (ecfp4 + count)


def _build_context(
    x_query,
    x_train,
    y_train,
    feature_weights,
    selection_similarity,
    kernel_similarity,
    max_m=CONFIG.context_m,
):#为查询分子预先构建局部计算所需的上下文，避免η和κ扫描时重复提取数据。
    count = min(max_m, x_train.shape[0])
    positions = np.argpartition(-selection_similarity, count - 1, axis=1)[:, :count]
    selected = np.take_along_axis(selection_similarity, positions, axis=1)
    order = np.argsort(-selected, axis=1)
    positions = np.take_along_axis(positions, order, axis=1)
    kernel_values = np.take_along_axis(kernel_similarity, positions, axis=1)

    weighted = np.zeros(
        (x_query.shape[0], count, y_train.shape[1]), dtype=float
    )
    for label in range(y_train.shape[1]):
        weights = feature_weights[:, label]
        if not np.any(weights):
            weights = np.ones_like(weights)
        weighted_query = x_query.multiply(weights)
        intersection = (weighted_query @ x_train.T).toarray()
        union = (
            np.asarray(weighted_query.sum(axis=1)).ravel()[:, None]
            + np.asarray(x_train @ weights).ravel()[None, :]
            - intersection
        )
        similarity = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        weighted[:, :, label] = np.take_along_axis(
            similarity, positions, axis=1
        )

    molecule_sizes = np.asarray(x_train.sum(axis=1)).ravel()
    active_features = []
    candidate_features = []
    candidate_sizes = []
    for query in range(x_query.shape[0]):
        active = x_query.indices[x_query.indptr[query] : x_query.indptr[query + 1]]
        candidates = positions[query]
        active_features.append(active)
        candidate_features.append(x_train[candidates][:, active])
        candidate_sizes.append(molecule_sizes[candidates])
    return {
        "positions": positions,
        "ecfp": kernel_values,
        "weighted": weighted,
        "active_features": active_features,
        "candidate_features": candidate_features,
        "candidate_sizes": candidate_sizes,
    }


def _kernels(
    context,
    eta,
    m=CONFIG.candidate_m,
    k=CONFIG.top_k,
    *,
    equal=False,
):#计算标签特异混合相似度并选邻居
    n_labels = context["weighted"].shape[2]
    eta = np.broadcast_to(np.asarray(eta, dtype=float), (n_labels,))
    hybrid = context["ecfp"][:, :, None] * (
        (1.0 - eta)[None, None, :]
        + eta[None, None, :] * context["weighted"]
    )
    result = np.zeros_like(hybrid)
    available = min(m, hybrid.shape[1])
    selected_count = min(k, available)
    for label in range(n_labels):
        values = hybrid[:, :available, label]
        selected = np.argpartition(-values, selected_count - 1, axis=1)[
            :, :selected_count
        ]
        selected_values = np.take_along_axis(values, selected, axis=1)
        if equal:
            selected_values = (selected_values > 0).astype(float)
        np.put_along_axis(
            result[:, :available, label], selected, selected_values, axis=1
        )
    return result


def _local_raw_scores(
    x_train,
    y_train,
    x_query,
    global_weights,
    context,
    kernels,
    kappa,
):#修改reverse
    global_forward, global_reverse = global_weights
    forward_raw = np.asarray(x_query @ global_forward, dtype=float).copy()
    reverse_raw = np.asarray(x_query @ global_reverse, dtype=float).copy()
    y_train = np.asarray(y_train, dtype=float)
    kappa = np.broadcast_to(np.asarray(kappa, dtype=float), (y_train.shape[1],))
    for query in range(x_query.shape[0]):
        active = context["active_features"][query]
        if active.size == 0:
            continue
        candidates = context["positions"][query]
        candidate_features = context["candidate_features"][query]
        candidate_sizes = context["candidate_sizes"][query]
        for label in range(y_train.shape[1]):
            kernel = kernels[query, :, label]
            if not np.any(kernel):
                continue
            support = np.asarray(kernel @ candidate_features).ravel()
            positive_kernel = kernel * y_train[candidates, label]
            joint = np.asarray(positive_kernel @ candidate_features).ravel()
            positive_total = float(positive_kernel @ candidate_sizes)
            conditional = np.divide(
                joint,
                support,
                out=np.zeros_like(joint),
                where=support > 0,
            )
            term_frequency = (
                joint / positive_total
                if positive_total > 0
                else np.zeros_like(joint)
            )
            local_reverse = conditional * term_frequency
            reliability = support / (support + kappa[label])
            reverse_raw[query, label] += np.sum(
                reliability * (local_reverse - global_reverse[active, label])
            )
    return forward_raw, reverse_raw


def _prediction(forward_raw, reverse_raw, alpha):
    forward = normalize_rows(forward_raw)
    reverse = normalize_rows(reverse_raw)
    return distribution_rows((1.0 - alpha) * forward + alpha * reverse)


def _fit_alpha(forward_raw, reverse_raw, y):
    forward = normalize_rows(forward_raw)
    reverse = normalize_rows(reverse_raw)
    alpha = learn_alpha(forward, reverse, normalized_targets(y))
    return alpha, _prediction(forward_raw, reverse_raw, alpha)


def _fit_alpha_crossentropy(forward_raw, reverse_raw, y):
    forward = normalize_rows(forward_raw)
    reverse = normalize_rows(reverse_raw)
    alpha = learn_alpha_crossentropy(forward, reverse, normalized_targets(y))
    return alpha, _prediction(forward_raw, reverse_raw, alpha)


def _loss(y, prediction):
    target = distribution_rows(y)
    return float(-np.sum(target * np.log(np.clip(prediction, 1e-12, 1.0))))


def _select_eta(forward_oof, reverse_by_eta, y):
    candidates = []
    for eta, reverse in zip(ETA_VALUES, reverse_by_eta):
        alpha, prediction = _fit_alpha(forward_oof, reverse, y)
        objective = _loss(y, prediction) + CONFIG.eta_shrink * eta**2
        candidates.append((objective, float(eta), float(alpha)))
    return min(candidates)[1]


def fit(x, y, caches, train_indices, seed, *, task):
    if task not in {"top1", "multilabel"}:
        raise ValueError(f"Unknown prediction task: {task}")
    train_indices = np.asarray(train_indices)
    full_y = np.asarray(y[train_indices], dtype=float)
    forward_oof = np.zeros_like(full_y)
    baseline_reverse_by_eta = [np.zeros_like(full_y) for _ in ETA_VALUES]
    payloads = []
    folds = KFold(CONFIG.inner_splits, shuffle=True, random_state=seed)
    for inner_train, validation in folds.split(train_indices):
        global_train = train_indices[inner_train]
        global_validation = train_indices[validation]
        train_x = x[global_train]
        validation_x = x[global_validation]
        fold_y = np.asarray(y[global_train], dtype=float)
        weights = fit_weights(train_x, fold_y)
        label_feature_weights = feature_weights(train_x, fold_y, CONFIG)
        ecfp4, fused = _similarities(caches, global_validation, global_train)
        baseline_context = _build_context(
            validation_x,
            train_x,
            fold_y,
            label_feature_weights,
            ecfp4,
            ecfp4,
        )
        fused_context = _build_context(
            validation_x,
            train_x,
            fold_y,
            label_feature_weights,
            fused,
            fused,
        )
        forward_oof[validation] = np.asarray(
            validation_x @ weights[0], dtype=float
        )
        for eta_index, eta in enumerate(ETA_VALUES):
            kernels = _kernels(baseline_context, eta)
            _, baseline_reverse_by_eta[eta_index][validation] = _local_raw_scores(
                train_x,
                fold_y,
                validation_x,
                weights,
                baseline_context,
                kernels,
                10.0,
            )
        payloads.append(
            (inner_train, validation, weights, baseline_context, fused_context)
        )

    eta = _select_eta(forward_oof, baseline_reverse_by_eta, full_y)
    equal = False
    # Top-1与多标签共用连续权重和融合邻域上的OOF收缩强度选择。
    # 这里在最终使用的bit/count融合邻域上搜索，测试标签不参与选择。
    kappa_candidates = []
    fused_kernels = [
        _kernels(fused_context, eta, equal=False)
        for _, _, _, _, fused_context in payloads
    ]
    for kappa_value in KAPPA_VALUES:
        reverse = np.zeros_like(full_y)
        for payload, kernels in zip(payloads, fused_kernels):
            inner_train, validation, weights, _, fused_context = payload
            train_x = x[train_indices[inner_train]]
            fold_y = np.asarray(y[train_indices[inner_train]], dtype=float)
            validation_x = x[train_indices[validation]]
            _, reverse[validation] = _local_raw_scores(
                train_x,
                fold_y,
                validation_x,
                weights,
                fused_context,
                kernels,
                kappa_value,
            )
        alpha_value, prediction = _fit_alpha(forward_oof, reverse, full_y)
        kappa_candidates.append(
            (_loss(full_y, prediction), float(kappa_value), float(alpha_value))
        )
    kappa = min(kappa_candidates)[1]

    fused_reverse = np.zeros_like(full_y)
    for inner_train, validation, weights, _, fused_context in payloads:
        train_x = x[train_indices[inner_train]]
        fold_y = np.asarray(y[train_indices[inner_train]], dtype=float)
        validation_x = x[train_indices[validation]]
        kernels = _kernels(fused_context, eta, equal=equal)
        _, fused_reverse[validation] = _local_raw_scores(
            train_x,
            fold_y,
            validation_x,
            weights,
            fused_context,
            kernels,
            kappa,
        )
    # η与κ沿用已验证的OOF选择过程；最终融合α使用与多标签概率
    # 目标一致的一维有界交叉熵优化，不需要学习率或迭代轮数。
    alpha, oof_scores = _fit_alpha_crossentropy(
        forward_oof, fused_reverse, full_y
    )
    train_x = x[train_indices]
    model = {
        "train_indices": train_indices,
        "x": train_x,
        "y": full_y,
        "weights": fit_weights(train_x, full_y),
        "feature_weights": feature_weights(train_x, full_y, CONFIG),
        "eta": eta,
        "kappa": kappa,
        "alpha": alpha,
        "equal": equal,
    }
    if task == "multilabel":
        model["thresholds"] = learn_label_thresholds(full_y, oof_scores)
    return model


def predict(model, x, caches, query_indices):#使用训练好的模型预测新分子
    query_indices = np.asarray(query_indices)
    train_indices = model["train_indices"]
    _, fused = _similarities(caches, query_indices, train_indices)
    x_query = x[query_indices]
    context = _build_context(
        x_query,
        model["x"],
        model["y"],
        model["feature_weights"],
        fused,
        fused,
    )
    kernels = _kernels(context, model["eta"], equal=model["equal"])
    forward, reverse = _local_raw_scores(
        model["x"],
        model["y"],
        x_query,
        model["weights"],
        context,
        kernels,
        model["kappa"],
    )
    return _prediction(forward, reverse, model["alpha"])
