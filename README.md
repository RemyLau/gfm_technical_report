# Genomic foundation model exploration technical report

## Installation

```bash
conda create -n gfm python=3.11 -y && conda activate gfm

# Install PyTorch (adjust to your CUDA version accordingly)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install DNABert
cd DNABERT && pip install -e . && cd -

# Install reset of the dependencies
pip install -r requirements -U
```

## Reproducing experiments

### Preparing data

#### From archive

All processed data can be downloaded from this Zenodo link:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10701018.svg)](https://doi.org/10.5281/zenodo.10701018)

```bash
wget https://zenodo.org/records/10701018/files/processed.zip?download=1 -O data/processed.zip
unzip data/processed.zip -d data/
```

#### Process from scratch

If you would like to process all data from scratch, start by runing the the two
notebooks listed below, which downloads the DHSs data and extract exclusive
peaks along with their corresponding class labels. The two notebooks can take
up to an hour to run depending on your internet connection and compute resource.

- `notebooks/master_dataset.ipynb`
- `notebooks/filter_master.ipynb`

Next, run the processing scripts to convert the sequence into k-mer features and
other numerical features.

```bash
sh run/prepare_data.sh
```

### Runing experiments

```bash
sh run/run_all_exps.sh
```
