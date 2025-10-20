import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging

from torch.utils.data import DataLoader

from model import MMLBackbone
from src.dataset.adni_dataset import ADNIDataset
from src.dataset.eicu_dataset import eICUDataset
from src.dataset.mimic4_dataset import MIMIC4Dataset
from src.dataset.padufes_dataset import PADUFESDataset
from src.dataset.yelp_dataset import YELPDataset
from src.dataset.utils import mimic4_collate_fn, eicu_collate_fn
from src.helper import Helper
from src.utils import count_parameters


def parse_arguments(parser):
    # parser.add_argument("--dataset", type=str, default="mimic4")
    # parser.add_argument("--dataset", type=str, default="eicu")
    # parser.add_argument("--task", type=str, default="readmission")
    # parser.add_argument("--monitor", type=str, default="pr_auc")
    parser.add_argument("--dataset", type=str, default="adni")
    parser.add_argument("--task", type=str, default="y")
    parser.add_argument("--monitor", type=str, default="auc_macro_ovo")
    parser.add_argument("--dev", action="store_true", default=False)
    parser.add_argument("--load_no_label", type=bool, default=False)
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--code_pretrained_embedding", type=bool, default=True)
    parser.add_argument("--code_layers", type=int, default=2)
    parser.add_argument("--code_heads", type=int, default=2)
    parser.add_argument("--bert_type", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--rnn_layers", type=int, default=1)
    parser.add_argument("--rnn_type", type=str, default="GRU")
    parser.add_argument('--rnn_bidirectional', action='store_true')
    parser.add_argument('--no_rnn_bidirectional', dest='rnn_bidirectional', action='store_false')
    parser.set_defaults(rnn_bidirectional=False)
    parser.add_argument("--ffn_layers", type=int, default=2)
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--gnn_norm", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--monitor_criterion", type=str, default="max")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no_train", type=bool, default=False)
    parser.add_argument("--note", type=str, default="mml_v19")
    parser.add_argument("--exp_name_attr", type=list, default=["dataset", "task", "note"])
    parser.add_argument("--official_run", action="store_true", default=True)
    parser.add_argument("--no_cuda", type=bool, default=False)
    # new options for cleaner console & periodic eval
    parser.add_argument("--log_mode", type=str, default="minimal", choices=["minimal", "full"],
                        help="Console log mode: minimal shows only key metrics; full shows all.")
    parser.add_argument("--eval_every", type=int, default=1, help="Run val/test every N epochs (default 1)")
    args = parser.parse_args()
    return args

def run_entry():
    helper = Helper(parse_arguments)
    args = helper.args

    if args.dataset == "eicu":
        train_set = eICUDataset(split="train", task=args.task, dev=args.dev, load_no_label=args.load_no_label)
        val_set = eICUDataset(split="val", task=args.task)
        test_set = eICUDataset(split="test", task=args.task)
        args.num_classes = 1
        collate_fn = eicu_collate_fn
        tokenizer = train_set.tokenizer
    elif args.dataset == "mimic4":
        train_set = MIMIC4Dataset(split="train", task=args.task, dev=args.dev, load_no_label=args.load_no_label)
        val_set = MIMIC4Dataset(split="val", task=args.task)
        test_set = MIMIC4Dataset(split="test", task=args.task)
        args.num_classes = 1
        collate_fn = mimic4_collate_fn
        tokenizer = train_set.tokenizer
    elif args.dataset == "adni":
        train_set = ADNIDataset(split="train", task=args.task, dev=args.dev, load_no_label=args.load_no_label)
        val_set = ADNIDataset(split="val", task=args.task)
        test_set = ADNIDataset(split="test", task=args.task)
        args.num_classes = 3
        collate_fn = None
        tokenizer = None
    elif args.dataset == "padufes":
        # Create full splits; we'll handle dev subsampling proportionally below
        train_set = PADUFESDataset(split="train", dev=False, load_no_label=args.load_no_label)
        val_set = PADUFESDataset(split="val", dev=False)
        test_set = PADUFESDataset(split="test", dev=False)
        # infer classes from dataset mapping
        args.num_classes = len(train_set.label2idx)
        collate_fn = None
        tokenizer = None
    elif args.dataset == "yelp":
        # Create full splits; we'll handle dev subsampling proportionally below
        train_set = YELPDataset(split="train", dev=False)
        val_set = YELPDataset(split="val", dev=False)
        test_set = YELPDataset(split="test", dev=False)
        args.num_classes = len(train_set.label2idx)
        collate_fn = None
        tokenizer = None
    else:
        raise ValueError("Dataset not supported!")

    # If dev mode, proportionally subsample each split to preserve original ratio (e.g., ~8:1:1)
    if args.dev:
        from torch.utils.data import Subset
        import math

        n_train, n_val, n_test = len(train_set), len(val_set), len(test_set)
        # Choose per-dataset caps for train (preserve existing speed expectations)
        if args.dataset == "padufes":
            cap_train = 512
        elif args.dataset == "yelp":
            cap_train = 2000
        else:
            cap_train = n_train

        scale = min(1.0, cap_train / max(1, n_train))
        new_n_train = min(n_train, int(math.floor(n_train * scale)))
        new_n_val = min(n_val, max(1, int(math.floor(n_val * scale))))
        new_n_test = min(n_test, max(1, int(math.floor(n_test * scale))))

        train_set = Subset(train_set, list(range(new_n_train)))
        val_set = Subset(val_set, list(range(new_n_val)))
        test_set = Subset(test_set, list(range(new_n_test)))

    # Safer defaults for Windows when using multiprocessing
    import platform
    default_workers = 4 if args.official_run else 0
    if platform.system() == "Windows":
        default_workers = 0 if not args.dev else 0

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=default_workers,
        pin_memory=True,
        shuffle=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=default_workers,
        pin_memory=True,
        shuffle=False
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=default_workers,
        pin_memory=True,
        shuffle=False
    )

    model = MMLBackbone(args, tokenizer)
    model.to(args.device)
    # Only print full model in full log mode
    if args.log_mode == "full":
        logging.info(model)
    param_count = count_parameters(model)
    logging.info("Number of parameters: {}".format(param_count))
    logging.info(
        f"Dataset sizes: train={len(train_set)} val={len(val_set)} test={len(test_set)}"
    )
    # Concise run summary for minimal console output
    logging.info(
        f"Run summary: model={model.model.__class__.__name__} dataset={args.dataset} "
        f"task={getattr(args, 'task', '')} epochs={args.epochs} batch_size={args.batch_size} "
        f"params={param_count}"
    )

    if args.checkpoint:
        helper.load_checkpoint(model, args.checkpoint)

    if not args.no_train:
        for epoch in range(args.epochs):

            logging.info("-------train: {}-------".format(epoch))
            scores = model.train_epoch(train_loader)
            for key in scores:
                helper.log(f"metrics/train/{key}", scores[key])
            helper.save_checkpoint(model, "last.ckpt")

            # Validation every epoch (original behavior)
            logging.info("-------val: {}-------".format(epoch))
            scores, _ = model.eval_epoch(val_loader, bootstrap=False)
            for key in scores.keys():
                helper.log(f"metrics/val/{key}", scores[key])
            helper.save_checkpoint_if_best(model, "best.ckpt", scores)

            # Test periodically
            if (epoch + 1) % max(1, args.eval_every) == 0:
                logging.info("-------test: {}-------".format(epoch))
                scores, predictions = model.eval_epoch(test_loader, bootstrap=False)
                for key in scores.keys():
                    helper.log(f"metrics/test/{key}", scores[key])

            if not args.official_run:
                break

        best_path = os.path.join(helper.model_saved_path, "best.ckpt")
        if os.path.exists(best_path):
            helper.load_checkpoint(model, best_path)
        else:
            logging.info(f"# best checkpoint not found, skip loading: {best_path}")

    logging.info("-------final test-------")
    scores, predictions = model.eval_epoch(test_loader, bootstrap=True)
    for key in scores.keys():
        helper.log(f"metrics/final_test/{key}", scores[key])
    helper.save_predictions(predictions)


if __name__ == "__main__":
    run_entry()
