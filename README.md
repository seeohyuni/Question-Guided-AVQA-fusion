<div align="center">

# Question-Guided Audio-Visual Fusion

### 질의 연관 시청각 단서 정제 및 시공간 관계 추론 기반 Audio-Visual Question Answering

**SeoHyeon Park**

</div>


## 🙌 Get Started

### 1. Clone This Repo

```bash
git clone https://github.com/seeohyuni/Question-Guided-AVQA-fusion.git
cd Question-Guided-AVQA-fusion
```

This repository assumes that the user already has a working PyTorch environment.

### 2. Prepare Data

The evaluation code uses pre-extracted MUSIC-AVQA features. The JSON split files and model checkpoints are included in this repository, while the feature files should be downloaded separately and placed under `./features`.

Expected directory structure:

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

### 3. Evaluation

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

The evaluation results are saved as JSON files under `./checkpoints`.
