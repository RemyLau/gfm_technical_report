#!/bin/bash --login
#
# KMER=3 CUDA_VISIBLE_DEVICES=5 sh run/run_dnase_tissue.sh

SEED=${SEED:-0}
KMER=${KMER:-6}

HOMEDIR=$(dirname $(dirname $(realpath $0)))
PT_PATH=$HOMEDIR/models/pt/${KMER}-new-12w-0
FT_PATH=$HOMEDIR/models/ft/dnase-tissue

cd ${HOMEDIR}/scripts

MODEL_PATH=$PT_PATH
DATA_PATH=$HOMEDIR/data/processed/tissue_cls/$KMER
OUTPUT_PATH=$FT_PATH/k${KMER}/r${SEED}

python run_finetune.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnase-tissue \
    --do_train \
    --do_eval \
    --data_dir $DATA_PATH \
    --max_seq_length 200 \
    --per_gpu_eval_batch_size=32 \
    --per_gpu_train_batch_size=32 \
    --learning_rate 2e-4 \
    --num_train_epochs 2.0 \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 100 \
    --save_steps 4000 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --weight_decay 0.01 \
    --seed ${SEED} \
    --n_process 8
