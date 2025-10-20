# MUSE
MUSE baseline Reproduction

usage() {
  cat <<USAGE
Run PAD-UFES with adjustable options.

Usage: $(basename "$0") [options] [-- extra-args]

Options:
  -D <dataset>        Dataset name (default: ${DATASET})
  -E <epochs>         Epochs (default: ${EPOCHS})
  -B <batch_size>     Batch size (default: ${BATCH_SIZE})
  -V <eval_every>     Test every N epochs (default: ${EVAL_EVERY})
  -L <log_mode>       Console log mode: minimal|full (default: ${LOG_MODE})
  -l <lr>             Learning rate (default: ${LR})
  -W <weight_decay>   Weight decay (default: ${WEIGHT_DECAY})
  -p <dropout>        Dropout rate (default: ${DROPOUT})
  -f <ffn_layers>     FFN layers (default: ${FFN_LAYERS})
  -g <gnn_layers>     GNN layers (default: ${GNN_LAYERS})
  -s <seed>           Random seed (default: ${SEED})
  -M <monitor>        Monitor metric (default: ${MONITOR})
  -C <criterion>      Monitor criterion max|min (default: ${MONITOR_CRITERION})
  -n <note>           Note tag (default: ${NOTE})
  -c <checkpoint>     Path to checkpoint to load (optional)
  --no-dev            Do not pass --dev
  --no-train          Add --no_train True
  -h|--help           Show this help

USAGE
}

# Run
./missmodal/Scripts/python.exe main.py \
  --dataset padufes \
  --dev \
  --epochs 10 \
  --batch_size 16 \
  --eval_every 1 \
  --log_mode minimal \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --dropout 0.3 \
  --ffn_layers 2 \
  --gnn_layers 2 \
  --seed 42 \
  --monitor auc_macro_ovo \
  --monitor_criterion max \
  --note mml_v19

