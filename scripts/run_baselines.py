import os
import time
import warnings
from abc import ABC, abstractmethod
from functools import partial
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Literal, Optional, Tuple, Union

import click
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


def format_time(t: float):
    return time.strftime("%H:%M:%S", time.gmtime(t)) + f"{t % 1:.2f}"[1:]


class Baseline(ABC):

    DISPLAY_ARGS = ()

    def __init__(self, data_dir: Path, kmer: int, model_kwargs: Optional[Dict[str, Any]] = None):
        self.data_dir = data_dir
        self.kmer = kmer

        self._data = {}

        default_model_kwargs = dict(penalty="l2", C=1.0, solver="lbfgs", random_state=42)
        if model_kwargs is not None:
            default_model_kwargs.update(model_kwargs)
        self._model = LogisticRegression(**default_model_kwargs)

    def __repr__(self) -> str:
        args_str = ", ".join([f"{arg_name}={getattr(self, arg_name)!r}" for arg_name in self.DISPLAY_ARGS])
        return f"{self.__class__.__name__}({args_str})"

    @abstractmethod
    def feature_processor_fit_transform(self, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def process_feature(self, x: np.ndarray) -> np.ndarray:
        ...

    def train(self):
        x_train, y_train = self.get_data(split="train", skip_processing=True)
        x_train = self.feature_processor_fit_transform(x_train)
        self._model.fit(x_train, y_train.ravel())

    def eval(self, split: Literal["train", "val", "test"]) -> Dict[str, float]:
        x, y = self.get_data(split=split)
        pred = self._model.predict(x)
        return self.compute_metrics(y, pred, prefix=split)

    def full_eval_report(self, metadata: Optional[Dict[str, Any]] = None):
        print(f"Start evaluating {self!r}")

        results = metadata.copy() or {}

        t = time.perf_counter()
        self.train()
        print("Training done, start evaluating")
        results["time_train"] = format_time(time.perf_counter() - t)

        for split in ["train", "val", "test"]:
            results.update(self.eval(split=split))
        results["time_total"] = format_time(time.perf_counter() - t)

        print(f"Evaluation done, results are as follows:\n{pformat(results)}\n")

        return results

    def compute_metrics(self, y_true, y_pred, prefix) -> Dict[str, float]:
        return {
            f"{prefix}_acc": (y_pred == y_true).mean(),
        }

    def get_data(
        self,
        split: Literal["train", "val", "test"],
        skip_processing=False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if split not in self._data:
            path = self.data_dir / f"{split}.npz"
            npfile = np.load(path)
            print(f"Loading data from {path}")
            self._data[split] = (npfile["x"], npfile["y"])

        x, y = self._data[split]
        if not skip_processing:
            x = self.process_feature(x)

        return x, y


class BaselineSimple(Baseline):

    def feature_processor_fit_transform(self, x):
        return x

    def process_feature(self, x):
        return x


class BaselineFreqRaw(BaselineSimple):

    DISPLAY_ARGS = ("kmer", )

    def __init__(self, data_dir: Path, kmer: int, **kwargs):
        data_dir = data_dir / str(kmer) / "extracted_features" / "frequency"
        super().__init__(data_dir, kmer, **kwargs)


class BaselineRandom(BaselineFreqRaw):

    DISPLAY_ARGS = ()

    def train(self):
        _, y_train = self.get_data(split="train")
        freq = pd.DataFrame(y_train).value_counts().sort_index().values
        self._prob = freq / freq.sum()

    def eval(self, split: Literal["train", "val", "test"]) -> Dict[str, float]:
        _, y = self.get_data(split=split)
        pred = np.random.choice(self._prob.size, size=y.size, p=self._prob)
        return self.compute_metrics(y, pred, prefix=split)


class BaselineFreqTfidf(BaselineFreqRaw):

    def __init__(self, data_dir: Path, kmer: int, **kwargs):
        super().__init__(data_dir, kmer, **kwargs)
        self.tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)

    def feature_processor_fit_transform(self, x):
        return self.tfidf.fit_transform(x)

    def process_feature(self, x):
        return self.tfidf.transform(x)


class BaselineFreqTfidfSVD(BaselineFreqTfidf):

    DISPLAY_ARGS = ("kmer", "n_components")

    def __init__(self, data_dir: Path, kmer: int, n_components: int = 128, **kwargs):
        super().__init__(data_dir, kmer, **kwargs)
        self.n_components = n_components
        self.tsvd = TruncatedSVD(n_components=n_components)

    def feature_processor_fit_transform(self, x):
        if x.shape[1] < self.n_components:
            warnings.warn(
                f"Hidden dimension ({self.n_components}) exceeds feature dimension "
                f"({x.shape[1]}). Implicitly hidden dimension to {x.shape[1]}",
                RuntimeWarning,
                stacklevel=2,
            )
            self.n_components = x.shape[1]
            self.tsvd = TruncatedSVD(n_components=self.n_components)
        return self.tsvd.fit_transform(super().feature_processor_fit_transform(x))

    def process_feature(self, x):
        return self.tsvd.transform(super().process_feature(x))


class BaselineDNABert(BaselineSimple):

    DISPLAY_ARGS = ("kmer", "mode", "model_name_template")

    def __init__(
        self,
        data_dir: Path,
        kmer: int,
        mode: Literal["pooled", "avg"] = "pooled",
        model_name_template: str = "{}-new-12w-0",
        **kwargs,
    ):
        self.mode = mode
        self.model_name_template = model_name_template
        model_name = model_name_template.format(kmer) + "_" + mode
        data_dir = data_dir / str(kmer) / "extracted_features" / model_name
        super().__init__(data_dir, kmer, **kwargs)


def run_eval_pipeline(args) -> Dict[str, Any]:
    task, kmer, baseline_name = args
    data_dir = PROCESSED_DATADIR / TASK_DIR_NAMES[task]
    baseline_cls = BASELINES[baseline_name]

    baseline = baseline_cls(data_dir=data_dir, kmer=kmer)
    metadata = dict(method=baseline_name, task=task, kmer=kmer)
    out = baseline.full_eval_report(metadata=metadata)

    return out


HOMEDIR = Path(__file__).resolve().parents[1]
PROCESSED_DATADIR = HOMEDIR / "data" / "processed"
MODEL_NAME_TEMPLATE = "{}-new-12w-0"

BASELINES = {
    "random": BaselineRandom,
    "freq-raw": BaselineFreqRaw,
    "freq-tfidf": BaselineFreqTfidf,
    "freq-tfidf-svd": BaselineFreqTfidfSVD,
    "dnabert-pooled": partial(BaselineDNABert, mode="pooled", model_name_template=MODEL_NAME_TEMPLATE),
    "dnabert-avg": partial(BaselineDNABert, mode="avg", model_name_template=MODEL_NAME_TEMPLATE),
}
TASK_DIR_NAMES = {
    "celltype": "cell_type_cls",
    "tissue": "tissue_cls",
    "cancer": "cancer_kidney_tubular_cls",
}

KMERS = [3, 4, 5, 6]


@click.command()
@click.option("--kmer", default="all")
@click.option("--task", type=str, default="all")
@click.option("--baseline", type=str, default="all")
@click.option("--out_path", type=str, default=str(HOMEDIR / "results"))
@click.option("--out_file_name", type=str, default="baseline_results")
@click.option("--n_jobs", type=int, default=1)
def main(
    kmer: Union[int, str],
    task: str,
    baseline: str,
    out_path: str,
    out_file_name: str,
    n_jobs: int,
):
    # Prepare baseline evaluation jobs
    kmers = KMERS if kmer == "all" else [kmer]
    tasks = list(TASK_DIR_NAMES) if task == "all" else [kmer]
    baselines = list(BASELINES) if baseline == "all" else [baseline]
    jobs = list(product(tasks, kmers, baselines))

    # Launch basleine evaluations
    # WARNING: dnabert based baselines consume a lot of memory, be sure to not
    # set n_jobs too large, which could cause stalking processes due to memory
    # issues. Depending on the scheduling, n_jobs of two could use up to 1T of
    # memory. Other frequency-based feature baslines are much less memory
    # intensive.
    with Pool(n_jobs) as p:
        full_results = list(
            tqdm(
                p.imap(run_eval_pipeline, jobs),
                total=len(jobs),
                desc="Evaluating baslines",
            )
        )

    # Save results
    os.makedirs(out_path, exist_ok=True)
    timenow = time.strftime("%Y-%m-%d", time.gmtime())
    path = os.path.join(out_path, f"{out_file_name}_{timenow}.csv")

    df = pd.DataFrame(full_results)
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")


if __name__ == "__main__":
    main()
