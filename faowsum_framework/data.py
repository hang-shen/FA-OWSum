from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATASET_NUMBER = {"A": 1, "B": 2, "C": 3, "D": 4}
MULTILABEL_LABELS = (
    "winey",
    "rose",
    "caramellic",
    "earthy",
    "balsamic",
    "citrus",
    "pineapple",
    "burnt",
    "musty",
    "phenolic",
)
MULTILABEL20_LABELS = tuple(
    "pineapple pear banana peach grape creamy buttery honey vanilla coffee musk "
    "amber powdery jasmin violet phenolic onion burnt tobacco smoky".split()
)
MULTILABEL_DATASETS = ("ML10", "ML20")


def parse_descriptor_list(value):#将字符串解析为python列表
    return [
        item.strip().strip("'").strip('"')
        for item in str(value).strip("[]").split(", ")
        if item.strip()
    ]


def _label_matrix(descriptors, labels):#将分子标签列表转换为0/1矩阵
    label_index = {label: index for index, label in enumerate(labels)}
    result = np.zeros((len(descriptors), len(labels)), dtype=np.float64)
    for row, values in enumerate(descriptors):
        for value in values:
            if value in label_index:
                result[row, label_index[value]] = 1.0
    return result


def load_top1_dataset(dataset):
#读取SMARTS特征；
#读取SMILES和气味标签；
#检查特征、标签行数是否一致；
#确定标签顺序；
#将标签转换为多热矩阵；
#将SMILES转换成RDKit分子
    dataset = dataset.upper()
    number = DATASET_NUMBER[dataset]
    suffix = f"GSLF_6labels_sub10_300_{number}.csv"
    features = pd.read_csv(
        DATA / "top1" / f"molecular_nonzero_{suffix}", index_col="index"
    )
    labels_frame = pd.read_csv(
        DATA / "top1" / f"odor_file_{suffix}",
        converters={"descriptor": parse_descriptor_list},
    )
    if len(features) != len(labels_frame):
        raise ValueError(f"Feature and label row counts differ for dataset {dataset}")
    labels = tuple(
        sorted({label for values in labels_frame.descriptor for label in values})
    )
    smiles = labels_frame["smiles"].to_numpy(dtype=object)
    molecules = [Chem.MolFromSmiles(value) for value in smiles]
    if any(molecule is None for molecule in molecules):
        raise ValueError(f"Invalid SMILES found in Top1 dataset {dataset}")
    x = sparse.csr_matrix((features.to_numpy() > 0).astype(np.float64))
    y = _label_matrix(labels_frame.descriptor, labels)
    return features.columns.to_numpy(dtype=object), x, y, smiles, molecules, labels


def load_multilabel_dataset(dataset="ML10"):
    dataset = dataset.upper()
    if dataset == "ML20":
        return load_multilabel20_dataset()
    if dataset != "ML10":
        raise ValueError(f"Unknown multilabel dataset: {dataset}")
    features = pd.read_csv(
        DATA / "multilabel" / "merged_AB_10labels_features.csv",
        index_col="smiles",
    )
    labels_frame = pd.read_csv(
        DATA / "multilabel" / "merged_AB_10labels_labels.csv",
        converters={"descriptor": parse_descriptor_list},
        index_col="smiles",
    )
    if not features.index.equals(labels_frame.index):
        raise ValueError("Multilabel feature and label SMILES rows are not aligned")
    smiles = features.index.to_numpy(dtype=object)
    molecules = [Chem.MolFromSmiles(value) for value in smiles]
    if any(molecule is None for molecule in molecules):
        raise ValueError("Invalid SMILES found in multilabel dataset")
    x = sparse.csr_matrix((features.to_numpy() > 0).astype(np.float64))
    y = _label_matrix(labels_frame.descriptor, MULTILABEL_LABELS)
    return (
        features.columns.to_numpy(dtype=object),
        x,
        y,
        smiles,
        molecules,
        MULTILABEL_LABELS,
    )


def load_multilabel20_dataset():
    folder = DATA / "二十标签完整标注数据集"
    source = folder / "dataset.csv"
    frame = pd.read_csv(source)
    if len(frame) != 2703 or not frame.sample_id.is_unique or not frame.smiles.is_unique:
        raise ValueError("ML20 must contain 2703 unique sample IDs and SMILES")
    y = frame.loc[:, MULTILABEL20_LABELS].to_numpy(dtype=np.float64)
    if not np.isin(y, [0, 1]).all() or not np.array_equal(y.sum(axis=1), frame.label_count):
        raise ValueError("ML20 labels must be binary and match label_count")
    if np.any(y.sum(axis=1) == 0) or y.sum() != 4072:
        raise ValueError("ML20 annotation counts differ from the frozen dataset")
    descriptors = _label_matrix(frame.descriptor.map(parse_descriptor_list), MULTILABEL20_LABELS)
    if not np.array_equal(descriptors, y):
        raise ValueError("ML20 descriptors and binary labels differ")
    smiles = frame.smiles.to_numpy(dtype=object)
    molecules = [Chem.MolFromSmiles(value) for value in smiles]
    if any(molecule is None for molecule in molecules):
        raise ValueError("Invalid SMILES in ML20")
    vocabulary = pd.read_csv(
        DATA / "multilabel/merged_AB_10labels_features.csv", nrows=0, index_col="smiles"
    ).columns.tolist()
    if len(vocabulary) != 4525 or len(set(vocabulary)) != len(vocabulary):
        raise ValueError("Expected the retained 4525-item SMARTS vocabulary")
    identity = {
        "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "smarts_patterns_sha256": hashlib.sha256(
            json.dumps(vocabulary, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "labels": list(MULTILABEL20_LABELS),
        "sample_ids": frame.sample_id.tolist(),
    }
    cache = folder / "inputs"
    matrix_file = cache / "smarts.npz"
    metadata_file = cache / "feature_manifest.json"
    valid_cache = False
    if matrix_file.exists() and metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        valid_cache = all(metadata.get(key) == value for key, value in identity.items())
        valid_cache = valid_cache and metadata.get("smarts_sha256") == hashlib.sha256(matrix_file.read_bytes()).hexdigest()
    if valid_cache:
        x = sparse.load_npz(matrix_file).astype(np.float64)
    else:
        patterns = [Chem.MolFromSmarts(pattern) for pattern in vocabulary]
        if any(pattern is None for pattern in patterns):
            raise ValueError("Invalid SMARTS in the retained vocabulary")
        print("Preparing ML20: 2703 molecules x 4525 frozen SMARTS patterns", flush=True)
        rows, columns = [], []
        for row, molecule in enumerate(molecules):
            hits = [column for column, pattern in enumerate(patterns) if molecule.HasSubstructMatch(pattern)]
            rows.extend([row] * len(hits))
            columns.extend(hits)
        x = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float64), (rows, columns)),
            shape=(len(frame), len(vocabulary)),
        )
        cache.mkdir(parents=True, exist_ok=True)
        temporary = cache / "smarts.tmp.npz"
        sparse.save_npz(temporary, x)
        temporary.replace(matrix_file)
        metadata = dict(identity, smarts_sha256=hashlib.sha256(matrix_file.read_bytes()).hexdigest())
        for filename, value in (("labels.json", list(MULTILABEL20_LABELS)), ("smarts_patterns.json", vocabulary), ("feature_manifest.json", metadata)):
            (cache / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    if x.shape != (len(frame), len(vocabulary)) or not np.isin(x.data, [0, 1]).all():
        raise ValueError("ML20 SMARTS matrix has invalid dimensions or values")
    return np.asarray(vocabulary, dtype=object), x, y, smiles, molecules, MULTILABEL20_LABELS
