#!/bin/bash --login
#
# TASK=dnase-tissue KMER=3 CUDA_VISIBLE_DEVICES=5 sh run/extract_features.sh

TASK=${TASK:-dnase-tissue}  # dnase-tissue | dnase-celltype | dnase-cancer
SEED=${SEED:-0}
KMER=${KMER:-6}

HOMEDIR=$(dirname $(dirname $(realpath $0)))
PT_PATH=$HOMEDIR/models/pt/${KMER}-new-12w-0
FT_PATH=$HOMEDIR/models/ft/${TASK}

cd ${HOMEDIR}/scripts

MODEL_PATH=$PT_PATH
if [[ $TASK == dnase-tissue ]]; then
    DATA_PATH=$HOMEDIR/data/processed/tissue_cls/$KMER
elif [[ $TASK == dnase-celltype ]]; then
    DATA_PATH=$HOMEDIR/data/processed/cell_type_cls/$KMER
elif [[ $TASK == dnase-cancer ]]; then
    DATA_PATH=$HOMEDIR/data/processed/cancer_kidney_tubular_cls/$KMER
else
    echo ERROR: unknown TASK=${TASK}
    exit
fi

python extract_features.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name ${TASK} \
    --data_dir $DATA_PATH \
    --max_seq_length 200 \
    --per_gpu_eval_batch_size=32 \
    --n_process 8
