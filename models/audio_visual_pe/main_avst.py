from __future__ import print_function

import argparse
import ast
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# modified: use the local dataloader/model for this experiment.
from dataloader_avst import AVQA_dataset, build_answer_vocab
from net_avst import AVQA_Fusion_Net

TASK_NAME_MAP = {
    ("Audio", "Counting"): "A_count",
    ("Audio", "Comparative"): "A_cmp",
    ("Visual", "Counting"): "V_count",
    ("Visual", "Location"): "V_loc",
    ("Audio-Visual", "Existential"): "AV_ext",
    ("Audio-Visual", "Counting"): "AV_count",
    ("Audio-Visual", "Location"): "AV_loc",
    ("Audio-Visual", "Comparative"): "AV_cmp",
    ("Audio-Visual", "Temporal"): "AV_temp",
}


def parse_question_type(raw_type):
    if isinstance(raw_type, str):
        return tuple(ast.literal_eval(raw_type))
    return tuple(raw_type)


def update_task_metrics(task_stats, sample_types, predicted, labels):
    for raw_type, pred, label in zip(sample_types, predicted.cpu(), labels.cpu()):
        parsed_type = parse_question_type(raw_type)
        task_key = TASK_NAME_MAP.get(parsed_type)
        group_key = "AV" if parsed_type[0] == "Audio-Visual" else parsed_type[0]
        for key in (task_key, group_key, "Overall"):
            if key is None:
                continue
            stats = task_stats.setdefault(key, {"correct": 0, "total": 0})
            stats["total"] += 1
            stats["correct"] += int(pred.item() == label.item())


def finalize_task_metrics(task_stats):
    return {
        key: {
            "accuracy": 100.0 * value["correct"] / value["total"],
            "correct": value["correct"],
            "total": value["total"],
        }
        for key, value in task_stats.items()
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audio-Visual Spatial-Temporal Model with CLIP/CLAP full AVQA features"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="/home/jun/Desktop/AVQA/feature/msclap_all",
        help="Directory containing CLAP audio feature .npy files.",
    )
    parser.add_argument(
        "--audio_fallback_dir",
        type=str,
        default="/home/jun/Desktop/AVQA/feature/msclap_av_counting",
        help="Fallback CLAP audio feature directory for samples not present in audio_dir.",
    )
    parser.add_argument(
        "--video_res14x14_dir",
        type=str,
        default="/home/jun/Desktop/AVQA/feature/clip_patch14_all",
        help="Directory containing CLIP patch14 visual feature .npy files.",
    )
    parser.add_argument(
        "--video_res14x14_fallback_dir",
        type=str,
        default="/home/jun/Desktop/AVQA/feature/clip_patch14_av_counting",
        help="Fallback CLIP patch14 feature directory for samples not present in video_res14x14_dir.",
    )
    parser.add_argument(
        "--text_feat_dir",
        type=str,
        default="/home/jun/Desktop/AVQA/feature/clip_text_all",
        help="Directory containing CLIP text feature .npy files indexed by question_id.",
    )
    parser.add_argument(
        "--text_feat_fallback_dir",
        type=str,
        default="/home/jun/Desktop/AVQA/feature/clip_text_av_counting/per_question",
        help="Fallback CLIP text feature directory for samples not present in text_feat_dir.",
    )
    parser.add_argument(
        "--label_train",
        type=str,
        default="/home/jun/Desktop/AVQA/dataset/json/old_data/json/avqa-train.json",
        help="Training label json.",
    )
    parser.add_argument(
        "--label_val",
        type=str,
        default="/home/jun/Desktop/AVQA/dataset/json/old_data/json/avqa-val.json",
        help="Validation label json.",
    )
    parser.add_argument(
        "--label_test",
        type=str,
        default="/home/jun/Desktop/AVQA/dataset/json/old_data/json/avqa-test.json",
        help="Test label json.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--max_time_steps", type=int, default=512)
    parser.add_argument("--positional_dropout", type=float, default=0.0)
    parser.add_argument(
        "--target_type",
        type=str,
        default="all",
        help='Use "all" for the full AVQA dataset, or a value like "Audio-Visual,Counting".',
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--sample_rate", type=int, default=6)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Stop training if validation accuracy does not improve for this many epochs.",
    )
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "train_test"],
        default="train_test",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default=str(CURRENT_DIR / "checkpoints"),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="avst_vq_channel_mlp_clipclap_all_data_audio_visual_sincos_pos_notrain",
    )
    return parser.parse_args()


def move_batch_to_device(sample, device):
    return {
        "audio": sample["audio"].to(device, non_blocking=True).float(),
        "visual_posi": sample["visual_posi"].to(device, non_blocking=True).float(),
        # modified: keep CLIP text feature dtype unchanged; net_avst uses it directly.
        "question": sample["question"].to(device, non_blocking=True),
        "label": sample["label"].to(device, non_blocking=True),
        "video_id": sample["video_id"],
        "question_id": sample["question_id"],
        "type": sample.get("type"),
    }


def make_dataset(label_path, args, mode_flag):
    return AVQA_dataset(
        label=label_path,
        audio_dir=args.audio_dir,
        video_res14x14_dir=args.video_res14x14_dir,
        text_feat_dir=args.text_feat_dir,
        audio_fallback_dir=args.audio_fallback_dir,
        video_res14x14_fallback_dir=args.video_res14x14_fallback_dir,
        text_feat_fallback_dir=args.text_feat_fallback_dir,
        answer_vocab_source=args.label_train,
        mode_flag=mode_flag,
        sample_rate=args.sample_rate,
        target_type=args.target_type,
    )


def make_loader(dataset, args, shuffle, device):
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": (device.type == "cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = True
    return DataLoader(**loader_kwargs)


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def train_one_epoch(args, model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    total_qa = 0
    correct_qa = 0
    total_loss = 0.0

    for batch_idx, sample in enumerate(tqdm(train_loader, desc=f"Train {epoch}", leave=False)):
        sample = move_batch_to_device(sample, device)
        audio = sample["audio"]
        visual_posi = sample["visual_posi"]
        target = sample["label"]
        question = sample["question"]

        optimizer.zero_grad(set_to_none=True)
        out_qa = model(audio, visual_posi, question)
        loss = criterion(out_qa, target)

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(out_qa.data, 1)
        total_qa += target.size(0)
        correct_qa += (predicted == target).sum().item()
        total_loss += loss.item() * target.size(0)

        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * target.size(0),
                    len(train_loader.dataset),
                    100.0 * batch_idx / max(len(train_loader), 1),
                    loss.item(),
                )
            )

    return total_loss / total_qa, 100.0 * correct_qa / total_qa


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    task_stats = {}

    for sample in tqdm(loader, desc=desc, leave=False):
        sample = move_batch_to_device(sample, device)
        preds_qa = model(
            sample["audio"],
            sample["visual_posi"],
            sample["question"],
        )
        loss = criterion(preds_qa, sample["label"])
        _, predicted = torch.max(preds_qa.data, 1)
        update_task_metrics(task_stats, sample["type"], predicted, sample["label"])

        batch_size = sample["label"].size(0)
        total += batch_size
        correct += (predicted == sample["label"]).sum().item()
        total_loss += loss.item() * batch_size

    return total_loss / total, 100.0 * correct / total, correct, total, finalize_task_metrics(task_stats)


def checkpoint_path(args):
    save_dir = Path(args.model_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir / f"{args.checkpoint}.pt"


def save_checkpoint(args, model, answer_vocab, best_val_acc):
    path = checkpoint_path(args)
    torch.save(
        {
            "model_state": unwrap_model(model).state_dict(),
            "answer_vocab": answer_vocab,
            "args": vars(args),
            "best_val_acc": best_val_acc,
        },
        path,
    )
    return path


def load_checkpoint(args, model, device):
    path = checkpoint_path(args)
    checkpoint = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint["model_state"])
    return checkpoint


def build_model(args, answer_vocab, device, train_dataset):
    sample = train_dataset[0]
    audio_dim = int(sample["audio"].shape[-1])
    model = AVQA_Fusion_Net(
        audio_dim=audio_dim,
        num_answers=len(answer_vocab),
        max_time_steps=args.max_time_steps,
        positional_dropout=args.positional_dropout,
    ).to(device)

    if device.type == "cuda" and torch.cuda.device_count() > 1 and "," in args.gpu:
        model = nn.DataParallel(model)

    print(
        f"Feature dims | audio={audio_dim} visual=512 text=512 | "
        f"answers={len(answer_vocab)} | target_type={args.target_type} | "
        f"samples={len(train_dataset)} | "
        f"positional_encoding=audio_visual_sinusoidal_notrain max_time_steps={args.max_time_steps}"
    )
    return model


def run_train(args, device):
    train_dataset = make_dataset(args.label_train, args, mode_flag="train")
    val_dataset = make_dataset(args.label_val, args, mode_flag="val")
    answer_vocab = build_answer_vocab(args.label_train, target_type=args.target_type)
    model = build_model(args, answer_vocab, device, train_dataset)

    train_loader = make_loader(train_dataset, args, shuffle=True, device=device)
    val_loader = make_loader(val_dataset, args, shuffle=False, device=device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    epochs_without_improvement = 0
    best_path = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            args, model, train_loader, optimizer, criterion, epoch, device
        )
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device, desc="Val")

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            best_path = save_checkpoint(args, model, answer_vocab, best_val_acc)
            print(f"Saved best checkpoint: {best_path}")
        else:
            epochs_without_improvement += 1
            print(
                "Early stopping counter: "
                f"{epochs_without_improvement}/{args.early_stopping_patience} "
                f"| best_val_acc={best_val_acc:.2f}"
            )
            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                print(
                    "Early stopping triggered | "
                    f"best_val_acc={best_val_acc:.2f} | "
                    f"patience={args.early_stopping_patience}"
                )
                break

    return best_path


def run_test(args, device):
    test_dataset = make_dataset(args.label_test, args, mode_flag="test")
    answer_vocab = build_answer_vocab(args.label_train, target_type=args.target_type)
    model = build_model(args, answer_vocab, device, test_dataset)
    load_checkpoint(args, model, device)

    test_loader = make_loader(test_dataset, args, shuffle=False, device=device)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, correct, total, task_metrics = evaluate(
        model, test_loader, criterion, device, desc="Test"
    )

    print(
        f"Test Accuracy: {test_acc:.2f} % "
        f"({correct}/{total}) | test_loss={test_loss:.4f}"
    )

    metrics = {
        "checkpoint": str(checkpoint_path(args)),
        "target_type": args.target_type,
        "positional_encoding": "audio_visual_sinusoidal_notrain",
        "max_time_steps": args.max_time_steps,
        "positional_dropout": args.positional_dropout,
        "accuracy": test_acc,
        "correct": correct,
        "total": total,
        "test_loss": test_loss,
        "task_metrics": task_metrics,
    }
    metrics_path = checkpoint_path(args).with_suffix(".test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved test metrics to: {metrics_path}")

    task_metrics_path = checkpoint_path(args).with_suffix(".task_test_metrics.json")
    with open(task_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved task test metrics to: {task_metrics_path}")


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # modified: experiment name reflects the added V-Q pre-spatial attention.
    print("\n--------------- CLIP/CLAP Full-Data VQ-Spatial AVST + Audio/Visual Sin/Cos Positional Encoding ---------------")
    print(f"Device: {device}")

    if args.mode in ("train", "train_test"):
        run_train(args, device)

    if args.mode in ("test", "train_test"):
        run_test(args, device)


if __name__ == "__main__":
    main()
