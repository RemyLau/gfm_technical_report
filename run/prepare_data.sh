#!/bin/bash --login

TASKS=(tissue celltype cancer)
KMERS=(3 4 5 6)

HOMEDIR=$(dirname $(realpath $0))
cd $HOMEDIR

# Download pre-trained DNABert models
mkdir -p ../models/pt/
gdown --fuzzy https://drive.google.com/file/d/1nVBaIoiJpnwQxiz4dSq6Sv9kBKfXhZuM/view?usp=sharing -O ../models/pt/
gdown --fuzzy https://drive.google.com/file/d/1V7CChcC6KgdJ7Gwdyn73OS6dZR_J-Lrs/view?usp=sharing -O ../models/pt/
gdown --fuzzy https://drive.google.com/file/d/1KMqgXYCzrrYD1qxdyNWnmUYPtrhQqRBM/view?usp=sharing -O ../models/pt/
gdown --fuzzy https://drive.google.com/file/d/1BJjqb5Dl2lNMg2warsFQ0-Xvn1xxfFXC/view?usp=sharing -O ../models/pt/
for kmer in ${KMERS[@]}; do
    unzip ../models/pt/${kmer}-new-12w-0.zip -d ../models/pt
done

# Process task- and kmer-specific data
for task in ${TASKS[@]}; do
    for kmer in ${KMERS[@]}; do
        # Prepare kmer data from sequence
        python ../scripts/process_data.py --file_path ../data/processed/tissue_cls/filtered_exclusive.ftr --n_jobs 8

        # Extract features using pretrained Bert (not finetuned)
        CUDA_VISIBLE_DEVICES=0 TASK=dnase-${task} KMER=${kmer} sh extract_features.sh
    done
done
