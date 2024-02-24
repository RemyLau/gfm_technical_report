#!/bin/bash --login

HOMEDIR=$(dirname $(realpath $0))
cd $HOMEDIR

TASKS=(tissue celltype cancer)
KMERS=(3 4 5 6)

# Run baselines
python ../scripts/run_baselines.py --n_jobs 1

# Evaluate DNABert fine-tuning
# seeds=(0 1 2 3 4)
seeds=(0)  # XXX: runing one seed for now due to limited computational resource
for seed in ${seed[@]}; do
    for task in ${TASKS[@]}; do
        for kmer in ${KMERS[@]}; do
            CUDA_VISIBLE_DEVICES=0 SEED=$seed KMER=${kmer} sh run_dnase_${task}.sh
            CUDA_VISIBLE_DEVICES=0 SEED=$seed KMER=${kmer} sh run_dnase_${task}.sh
            CUDA_VISIBLE_DEVICES=0 SEED=$seed KMER=${kmer} sh run_dnase_${task}.sh
        done
    done
done
