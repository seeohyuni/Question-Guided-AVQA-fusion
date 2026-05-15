import ast
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_TARGET_TYPE = None
SPLIT_BY_MODE = {
    "train": "avqa-train",
    "val": "avqa-val",
    "test": "avqa-test",
}


def _parse_type(sample_type):
    if isinstance(sample_type, str):
        return ast.literal_eval(sample_type)
    return sample_type


def _normalize_target_type(target_type):
    if target_type is None:
        return None
    if isinstance(target_type, str):
        lowered = target_type.strip().lower()
        if lowered in ("", "all", "none"):
            return None
        parsed = ast.literal_eval(target_type) if target_type.strip().startswith("[") else target_type
        if isinstance(parsed, str):
            return [part.strip() for part in parsed.split(",")]
        return parsed
    return target_type


def _normalize_sample(sample, target_type=DEFAULT_TARGET_TYPE):
    normalized = dict(sample)
    if "anser" not in normalized and "answer" in normalized:
        normalized["anser"] = normalized["answer"]
    if target_type is not None and ("type" not in normalized or normalized["type"] is None):
        normalized["type"] = target_type
    if "question_id" in normalized:
        normalized["question_id"] = int(normalized["question_id"])
    return normalized


def _matches_target_type(sample, target_type):
    if target_type is None:
        return True
    return _parse_type(sample.get("type")) == target_type


def _matches_split(sample, mode_flag):
    split = sample.get("split")
    if split is None:
        return True
    return split == SPLIT_BY_MODE.get(mode_flag, mode_flag)


def _as_path_list(*paths):
    return [Path(path) for path in paths if path]


def _resolve_text_dir(path):
    path = Path(path)
    per_question = path / "per_question"
    return per_question if per_question.is_dir() else path


def _first_existing_feature_path(directories, filename):
    checked = []
    for directory in directories:
        path = directory / filename
        checked.append(str(path))
        if path.exists():
            return path
    raise FileNotFoundError("Feature file not found. Checked: " + ", ".join(checked))


def build_answer_vocab(label_path, target_type=DEFAULT_TARGET_TYPE):
    target_type = _normalize_target_type(target_type)
    samples = json.load(open(label_path, "r"))
    answer_vocab = []
    for raw_sample in samples:
        sample = _normalize_sample(raw_sample, target_type=target_type)
        if not _matches_target_type(sample, target_type):
            continue
        answer = sample["anser"]
        if answer not in answer_vocab:
            answer_vocab.append(answer)
    return answer_vocab


class AVQA_dataset(Dataset):
    def __init__(
        self,
        label,
        audio_dir,
        video_res14x14_dir,
        text_feat_dir,
        answer_vocab_source=None,
        transform=None,
        mode_flag="train",
        sample_rate=6,
        target_type=DEFAULT_TARGET_TYPE,
        audio_fallback_dir=None,
        video_res14x14_fallback_dir=None,
        text_feat_fallback_dir=None,
    ):
        target_type = _normalize_target_type(target_type)
        raw_samples = json.load(open(label, "r"))
        self.samples = []
        for raw_sample in raw_samples:
            sample = _normalize_sample(raw_sample, target_type=target_type)
            if not _matches_target_type(sample, target_type):
                continue
            if not _matches_split(sample, mode_flag):
                continue
            self.samples.append(sample)
        if not self.samples:
            raise ValueError(f"No samples found for target type {target_type or 'all'} in {label}")

        vocab_source = answer_vocab_source or label
        self.ans_vocab = build_answer_vocab(vocab_source, target_type=target_type)
        self.answer_to_idx = {
            answer: idx for idx, answer in enumerate(self.ans_vocab)
        }

        self.audio_dirs = _as_path_list(audio_dir, audio_fallback_dir)
        self.video_res14x14_dirs = _as_path_list(video_res14x14_dir, video_res14x14_fallback_dir)
        self.text_feat_dirs = [_resolve_text_dir(path) for path in _as_path_list(text_feat_dir, text_feat_fallback_dir)]
        self.transform = transform
        self.mode_flag = mode_flag
        self.sample_rate = sample_rate
        self.target_type = target_type

    def __len__(self):
        return len(self.samples)

    def _load_audio_feature(self, video_id):
        audio_path = _first_existing_feature_path(self.audio_dirs, f"{video_id}.npy")
        audio = np.load(audio_path, mmap_mode="r")
        if audio.ndim != 2:
            raise ValueError(f"Unsupported audio feature shape for {audio_path}: {audio.shape}")
        return np.asarray(audio[::self.sample_rate], dtype=np.float32).copy()

    def _load_visual_feature(self, video_id):
        visual_path = _first_existing_feature_path(self.video_res14x14_dirs, f"{video_id}.npy")
        visual = np.load(visual_path, mmap_mode="r")

        if visual.ndim == 3:
            visual = visual[::self.sample_rate]
            time, patches, channels = visual.shape
            side = int(patches ** 0.5)
            if side * side != patches:
                raise ValueError(f"Cannot reshape visual patches for {visual_path}: {visual.shape}")
            visual = visual.reshape(time, side, side, channels).transpose(0, 3, 1, 2)
        elif visual.ndim == 4:
            visual = visual[::self.sample_rate]
            if visual.shape[-1] == 512 and visual.shape[1] != 512:
                visual = visual.transpose(0, 3, 1, 2)
        else:
            raise ValueError(f"Unsupported visual feature shape for {visual_path}: {visual.shape}")

        return np.asarray(visual, dtype=np.float32).copy()

    def _load_text_feature(self, question_id):
        text_path = _first_existing_feature_path(self.text_feat_dirs, f"{question_id}.npy")
        text = np.load(text_path).astype(np.float32)
        text = np.squeeze(text)
        if text.ndim != 1:
            raise ValueError(f"Unsupported text feature shape for {text_path}: {text.shape}")
        return text

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample["video_id"]
        question_id = sample["question_id"]

        audio = self._load_audio_feature(video_id)
        visual_posi = self._load_visual_feature(video_id)
        text = self._load_text_feature(question_id)

        if audio.shape[0] != visual_posi.shape[0]:
            min_time = min(audio.shape[0], visual_posi.shape[0])
            audio = audio[:min_time].copy()
            visual_posi = visual_posi[:min_time].copy()

        label = self.answer_to_idx[sample["anser"]]

        sample_dict = {
            "audio": torch.from_numpy(audio),
            "visual_posi": torch.from_numpy(visual_posi),
            "question": torch.from_numpy(text),
            "label": torch.tensor(label, dtype=torch.long),
            "question_id": torch.tensor(question_id, dtype=torch.long),
            "video_id": video_id,
            "type": sample["type"],
        }

        if self.transform:
            sample_dict = self.transform(sample_dict)

        return sample_dict


class ToTensor(object):
    def __call__(self, sample):
        return sample
