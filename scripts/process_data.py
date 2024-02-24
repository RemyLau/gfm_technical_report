"""Process filtered exclusive peaks labeled data into feature label pairs.

Example:
$ python process_data.py --file_path ../data/processed/tissue_cls/filtered_exclusive.ftr --n_jobs 32

"""
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional

import click
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

if (DEV_MODE := os.environ.get("DEV_MODE", False)):
    print("[WARNING] Runing in dev mode!!!")

MIN_SAMPLES_PER_CLASS = 200
TRAIN_RATE = 0.9
TEST_RATE = 0.05


def get_kmer_sequence(original_string, kmer=1):
    if kmer == -1:
        return original_string

    sequence = []
    original_string = original_string.replace("\n", "")
    for i in range(len(original_string) - kmer + 1):
        sequence.append(original_string[i:i+kmer])

    return sequence


def get_kmer_sequence_string(original_string, kmer: int = 1, sep: str = " "):
    return sep.join(get_kmer_sequence(original_string, kmer=kmer))


def get_stratified_split_dict(y):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=TRAIN_RATE, random_state=42)
    train_idx, val_test_idx = next(sss.split(y, y))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_RATE / (1 - TRAIN_RATE), random_state=42)
    val_idx, test_idx = next(sss.split(y[val_test_idx], y[val_test_idx]))
    val_idx, test_idx = val_test_idx[val_idx], val_test_idx[test_idx]

    assert np.intersect1d(train_idx, val_idx).size == 0
    assert np.intersect1d(train_idx, test_idx).size == 0
    assert np.intersect1d(test_idx, val_idx).size == 0

    return dict(train=train_idx, val=val_idx, test=test_idx)


def kmer_to_freq(kmer_feat, kmer_dict):
    freq_feat = np.zeros(len(kmer_dict))
    for i in kmer_feat:
        freq_feat[kmer_dict[i]] += 1
    return freq_feat


def get_frequency_features(kmer_feats: List[str], sep: str = " ", n_jobs: int = 1):
    kmer_feats_split = [i.split(sep) for i in kmer_feats]

    all_kmers = set()
    for kmers in tqdm(kmer_feats_split, desc="Obtaining kmer dict"):
        all_kmers.update(set(kmers))

    all_kmers = sorted(all_kmers)
    kmer_dict = {j: i for i, j in enumerate(all_kmers)}
    print(f"First 10 unique kmers (total {len(all_kmers):,}):\n{all_kmers[:10]}")

    with Pool(n_jobs) as p:
        freq_feats = list(
            tqdm(
                p.imap(partial(kmer_to_freq, kmer_dict=kmer_dict), kmer_feats_split),
                total=len(kmer_feats_split),
                desc="Extracting kmer frequency features",
            ),
        )
    freq_feat_mat = np.vstack(freq_feats)

    return all_kmers, freq_feat_mat


@click.command()
@click.option("--kmer", type=int, default=6)
@click.option("--file_path", type=str, required=True, help="Input processed feather dataset.")
@click.option("--out_path", type=str, default=None, help="Output path, set to parent of file_path if not specified.")
@click.option("--n_jobs", type=int, default=1, help="Number of parallel workers.")
def main(kmer: int, file_path: str, out_path: Optional[str], n_jobs: int):
    out_path = (Path(file_path).parent if out_path is None else Path(out_path)) / str(kmer)
    feat_out_path = out_path / "extracted_features" / "frequency"
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(feat_out_path, exist_ok=True)

    df = pd.read_feather(file_path)
    if DEV_MODE:
        df = df.sample(10000)

    # Convert sequences into kmers
    with Pool(n_jobs) as p:
        kmer_feats = list(
            tqdm(
                p.imap(
                    partial(get_kmer_sequence_string, kmer=kmer),
                    df["sequence"].tolist(),
                ),
                total=df.shape[0],
                desc="Extracting kmers",
            ),
        )
    df_processed = pd.DataFrame(dict(sequence=kmer_feats, label=df["TAG"].tolist()))

    # Filter out extreme rare classes
    class_stats = df_processed["label"].value_counts()
    print(f"Class stats:\n{df_processed['label'].value_counts()}")
    if (to_remove := class_stats[class_stats < MIN_SAMPLES_PER_CLASS].index.tolist()):
        print(f"Removing the following classes due to insufficient examples: {to_remove}")
        df_processed = df_processed[~df_processed["label"].isin(to_remove)].reset_index(drop=True)

    # Encode labels and save label mapping
    le = LabelEncoder()
    df_processed["label"] = le.fit_transform(df_processed["label"].tolist())
    with open(out_path / "label_order.txt", "w") as f:
        for i in le.classes_:
            f.write(f"{i}\n")
    print(f"Total number of classes: {le.classes_.size}")

    # Frequency features
    all_kmers, freq_feat_mat = get_frequency_features(kmer_feats, n_jobs=n_jobs)

    # Prepare splits and save individual files
    for split_name, split_idx in get_stratified_split_dict(df_processed["label"].values).items():
        save_path = out_path / f"{split_name}.tsv"
        df_processed.iloc[split_idx].to_csv(save_path, sep="\t", index=False)
        print(f"Saved file: {save_path}")

        save_path = feat_out_path / f"{split_name}.npz"
        np.savez_compressed(
            save_path,
            x=freq_feat_mat[split_idx],
            y=df_processed.iloc[split_idx, 1].tolist(),
            feat_names=all_kmers,
        )
        print(f"Saved extracted frequency features: {save_path}")


if __name__ == "__main__":
    main()
