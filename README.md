<div align="center">

# Question-Guided Audio-Visual Fusion

### 질의 연관 시청각 단서 정제 및 시공간 관계 추론 기반 Audio-Visual Question Answering

**SeoHyeon Park**

</div>


## Models

| Paper row | Directory | Checkpoint |
|---|---|---|
| Base | `models/base` | `checkpoints/base.pt` |
| Visual-only PE(V) | `models/visual_pe` | `checkpoints/visual_pe.pt` |
| Audio-only PE(A) | `models/audio_pe` | `checkpoints/audio_pe.pt` |
| Audio+Visual PE(V+A) | `models/audio_visual_pe` | `checkpoints/audio_visual_pe.pt` |

## Required Files

The repository expects the following structure:

```text
.
├── checkpoints/
│   ├── base.pt
│   ├── visual_pe.pt
│   ├── audio_pe.pt
│   └── audio_visual_pe.pt
├── dataset/json/
│   ├── avqa-train.json
│   ├── avqa-val.json
│   └── avqa-test.json
└── features/
    ├── msclap_all/
    ├── msclap_av_counting/
    ├── clip_patch14_all/
    ├── clip_patch14_av_counting/
    ├── clip_text_all/
    └── clip_text_av_counting/per_question/
```

The pre-extracted feature files are large and should be downloaded separately. Place them under `./features` before running evaluation.

## Environment

```bash
pip install -r requirements.txt
```

## Evaluation

Run one setting:

```bash
bash scripts/test_base.sh
bash scripts/test_visual_pe.sh
bash scripts/test_audio_pe.sh
bash scripts/test_audio_visual_pe.sh
```

Run all settings:

```bash
bash scripts/test_all.sh
```

If you want to reduce dataloader workers:

```bash
NUM_WORKERS=2 bash scripts/test_audio_pe.sh
```

## Expected Overall Accuracy

| Method | All Acc |
|---|---:|
| Base | 68.16 |
| Visual-only PE(V) | 69.66 |
| Audio-only PE(A) | 72.47 |
| Audio+Visual PE(V+A) | 70.27 |
