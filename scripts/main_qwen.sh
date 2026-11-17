#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --time=7-00:00:00
#SBATCH --account=kfragki2
#SBATCH --gpus-per-node=8
#SBATCH --mem-per-gpu=50G
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=L40|L40S|A6000|6000Ada
#SBATCH -o output/logs/%x_%j_%n.out
#SBATCH -e output/logs/%x_%j_%n.out
#SBATCH --requeue

printenv

get_cuda_device_count() {
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l
    else
        nvidia-smi --query-gpu=name --format=csv,noheader | wc -l
    fi
}

if [ "${DEBUGRUN:-0}" -eq 1 ]; then
    export IGNORERUN=1
    export CUDA_VISIBLE_DEVICES=0
    export NUM_DATALOADERS=${NUM_DATALOADERS:-0}
    export NUM_VAL_DATALOADERS=${NUM_VAL_DATALOADERS:-0}
    export BS=${BS:-2}
    export EVAL_PERIOD=${EVAL_PERIOD:-5}
    export NUM_GPUS=1
fi

if [ "${DEBUGFAST:-0}" -eq 1 ]; then
    export IGNORERUN=1
    export BS=${BS:-4}
    export EVAL_PERIOD=${EVAL_PERIOD:-30}
fi

DIR="$(dirname "$PWD")"
export PYTHONPATH=$DIR:$DIR/pretrain
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
# export TORCH_USE_CUDA_DSA=1  # Enable CUDA DSA for better performance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# export OMP_NUM_THREADS=8

export NUM_GPUS=${NUM_GPUS:-8}
export NUM_MACHINES=${SLURM_JOB_NUM_NODES:-1}
BS=${BS:-7}
SAMPLING_FRAME_NUM=${SAMPLING_FRAME_NUM:-15} # -15
SIDE_FRAMES=$(( (SAMPLING_FRAME_NUM - 1) / 2 ))
CHECKPOINT_PERIOD=${CHECKPOINT_PERIOD:-8000}
EVAL_PERIOD=${EVAL_PERIOD:-8000}
IGNORERUN=${IGNORERUN:-0}
NAME=${NAME:-"qwen3d"}
NUM_DATALOADERS=${NUM_DATALOADERS:-16}
NUM_VAL_DATALOADERS=${NUM_VAL_DATALOADERS:-4}
BREAKPOINT_ON_ERROR=${BREAKPOINT_ON_ERROR:-False}

RESUME=${RESUME:-1}

# babel paths (defaults; override any of these from the environment)
export DETECTRON2_DATASETS_2D="${DETECTRON2_DATASETS_2D:-/data/group_data/katefgroup-ssd/vlm/lucy/datasets_2d}"
export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-/data/user_data/lucylin/SEMSEG_100k}"
export SCANNET_PP_DATASET="${SCANNET_PP_DATASET:-/data/group_data/katefgroup/language_grounding/scannetpp/}"
export REF_DATASET="${REF_DATASET:-/data/user_data/ayushj2/refer_it_3d/}"
export LOCATE3D_DATASET="${LOCATE3D_DATASET:-/data/group_data/katefgroup-ssd/vlm/lucy/locate_3d}"
export COCO_REF_DATASET="${COCO_REF_DATASET:-/data/group_data/katefgroup-ssd/vlm/lucy/datasets_2d}"
export CLIP_SAMPLING_PATH="${CLIP_SAMPLING_PATH:-/data/user_data/ayushj2/refer_it_3d}"
export PRECOMPUTED_SCANNET_PATH="${PRECOMPUTED_SCANNET_PATH:-/data/user_data/ayushj2/refer_it_3d/}"
export LLAVA_INSTRUCT_PATH="${LLAVA_INSTRUCT_PATH:-/data/group_data/katefgroup-ssd/vlm/lucy/llava_instruct/llava_instruct_150k.json}"
MATTERPORT_DATA_DIR="${MATTERPORT_DATA_DIR:-/data/group_data/katefgroup-ssd/vlm/lucy/mask3d_processed/matterport/train_validation_database.yaml}"
SCANNET200_DATA_DIR="${SCANNET200_DATA_DIR:-/data/group_data/katefgroup-ssd/vlm/lucy/mask3d_processed/scannet200/train_validation_database.yaml}"
OUTPUT_DIR_PREFIX="${OUTPUT_DIR_PREFIX:-/data/group_data/katefgroup/language_grounding/qwen3d_outputs/train}"
SCANNET_DATA_DIR="${SCANNET_DATA_DIR:-/data/group_data/katefgroup-ssd/vlm/lucy/mask3d_processed/scannet/train_validation_database.yaml}"
S3DIS_DATA_DIR="${S3DIS_DATA_DIR:-/data/user_data/lucylin/SEMSEG_100k/s3dis/train_validation_database.yaml}"
SCANNETPP_DATA_DIR="${SCANNETPP_DATA_DIR:-/data/group_data/katefgroup-ssd/vlm/lucy/mask3d_processed/scannetpp/validation_database.yaml}"
FEATURE_DIR="${FEATURE_DIR:-/data/user_data/lucylin/scannet_image_qwen_features}"

echo "BS: ${BS}, NUM_GPUS: ${NUM_GPUS}, SAMPLING_FRAME_NUM: ${SAMPLING_FRAME_NUM}, NUM_DATALOADERS: ${NUM_DATALOADERS}, VAL: ${NUM_VAL_DATALOADERS}"

if [[ ("$NAME" = "qwen3d") && "$EVAL_ONLY" -eq 1 ]]; then
    NAME="${NAME}_eval"
fi

unset OUTPUT_DIR

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [[ -z "$OUTPUT_DIR" ]]; then
    if [[ "$IGNORERUN" -eq 1 ]]; then
        OUTPUT_DIR="${OUTPUT_DIR_PREFIX}/debug/ignore_${TIMESTAMP}_${NAME}"
        BREAKPOINT_ON_ERROR=True
    elif [[ "$NAME" = "qwen3d" || "$NAME" = "qwen3d_eval" ]]; then
        OUTPUT_DIR="${OUTPUT_DIR_PREFIX}/${TIMESTAMP}_${NAME}"
    else
        OUTPUT_DIR="${OUTPUT_DIR_PREFIX}/${NAME}"
    fi
    echo "Using output dir: ${OUTPUT_DIR}"
else
    echo "Using existing output dir: ${OUTPUT_DIR}"
fi




if [[ "$USE_SLURM" -eq 1 ]]; then
    SLURM_ARG="launcher=slurm"
    CMD="srun python"

    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_ADDR=$master_addr
    echo "MASTER_ADDR="$MASTER_ADDR
    export MASTER_PORT=$(shuf -n 1 -i 10000-65535)
    echo "MASTER_PORT="$MASTER_PORT

    export RANK=${SLURM_PROCID}
    echo "NODES="$NODES
    echo "RANK="$RANK
    echo "GPUS_PER_NODE="$GPUS_PER_NODE
    echo "SLURM_NODEID="$SLURM_NODEID
else
    SLURM_ARG=""
    CMD="python"
fi


CONFIG_FILE="qwen3d/configs/qwen_3d.yaml"


echo "EVAL_ONLY: $EVAL_ONLY RESUME: $RESUME"

if [[ "$EVAL_ONLY" -eq 1 ]]; then
    EVAL_ARG="--eval-only"
else
    EVAL_ARG=""
fi

if [[ "$RESUME" -eq 1 ]]; then
    RESUME_ARG="--resume"
else
    RESUME_ARG=""
fi

# Checks if CUDA is available on the node and if not, requeues job (if it detects we are inside a SLURM job).
python qwen3d/slurm_requeue.py

BS2D=${BS2D:-1}
BS3D=${BS3D:-1}

# Static training defaults live in qwen3d/config.py + qwen3d/configs/qwen_3d.yaml.
# Only runtime / path-derived overrides are passed here.
$CMD train.py --dist-url="tcp://127.0.0.1:$(shuf -i 19801-22947 -n 1)" --num-gpus $NUM_GPUS --num-machines $NUM_MACHINES --config-file $CONFIG_FILE $EVAL_ARG $RESUME_ARG $SLURM_ARG \
OUTPUT_DIR $OUTPUT_DIR \
SOLVER.IMS_PER_BATCH $((NUM_GPUS * NUM_MACHINES * BS)) \
SOLVER.CHECKPOINT_PERIOD $CHECKPOINT_PERIOD \
TEST.EVAL_PERIOD $EVAL_PERIOD \
INPUT.FRAME_LEFT $SIDE_FRAMES INPUT.FRAME_RIGHT $SIDE_FRAMES INPUT.SAMPLING_FRAME_NUM $SAMPLING_FRAME_NUM \
SOLVER.TEST_IMS_PER_BATCH $((NUM_GPUS * NUM_MACHINES)) \
DATALOADER.NUM_WORKERS $NUM_DATALOADERS \
DATALOADER.TEST_NUM_WORKERS $NUM_VAL_DATALOADERS \
SCANNET_DATA_DIR $SCANNET_DATA_DIR \
SCANNET200_DATA_DIR $SCANNET200_DATA_DIR \
MATTERPORT_DATA_DIR $MATTERPORT_DATA_DIR \
S3DIS_DATA_DIR $S3DIS_DATA_DIR \
SCANNETPP_DATA_DIR $SCANNETPP_DATA_DIR \
FEATURE_DIR $FEATURE_DIR \
SLURM_JOB_ID "\"${SLURM_JOB_ID:-}\"" \
SLURM_SUBMIT_DIR "\"${SLURM_SUBMIT_DIR:-}\"" \
SOLVER.IMS_PER_BATCH_2D $((NUM_GPUS * NUM_MACHINES * BS2D)) \
SOLVER.IMS_PER_BATCH_3D $((NUM_GPUS * NUM_MACHINES * BS3D)) \
VISUALIZE_LOG_DIR "$OUTPUT_DIR/viz_ref" \
TEST_RESULT_EXPORT_PATH "$OUTPUT_DIR/test_results" \
BREAKPOINT_ON_ERROR $BREAKPOINT_ON_ERROR \
$@
status=$?
if [[ $status == 124 ]]; then
  scontrol requeue $SLURM_JOB_ID
fi
exit $status
