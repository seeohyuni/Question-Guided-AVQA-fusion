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

The evaluation code uses pre-extracted MUSIC-AVQA features. The JSON split files and model checkpoints are included in this repository, while the feature files should be downloaded separately.

You can download and extract the feature archive with the following command. Hugging Face authentication may be required if the dataset repository is private or gated. 

```bash
python download_dataset.py
```

If `huggingface_hub` is not installed, install it first:

```bash
pip install huggingface_hub
```

If authentication is required, log in before running the download script:

```bash
huggingface-cli login
python download_dataset.py
```

The script downloads `features.zip` from Hugging Face and extracts it under `./features`.

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

## Results

Question type-wise accuracy comparison on the MUSIC-AVQA test set. The best result among our reproduced variants is highlighted in bold.

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="2">Audio</th>
      <th colspan="2">Visual</th>
      <th colspan="5">AV</th>
      <th rowspan="2">All<br>Acc</th>
    </tr>
    <tr>
      <th>Counting</th>
      <th>Cmp</th>
      <th>Counting</th>
      <th>Loc</th>
      <th>Ext</th>
      <th>Loc</th>
      <th>Counting</th>
      <th>Cmp</th>
      <th>Temporal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>AVST [3]</td>
      <td>77.30</td>
      <td>64.73</td>
      <td>72.64</td>
      <td>75.73</td>
      <td>82.41</td>
      <td>65.48</td>
      <td>69.94</td>
      <td>64.22</td>
      <td>64.44</td>
      <td>71.26</td>
    </tr>
    <tr>
      <td>AVSD [7]</td>
      <td>72.41</td>
      <td>61.90</td>
      <td>67.39</td>
      <td>74.19</td>
      <td>81.61</td>
      <td>58.79</td>
      <td>63.89</td>
      <td>61.52</td>
      <td>61.41</td>
      <td>67.44</td>
    </tr>
    <tr>
      <td>Pano [8]</td>
      <td>74.36</td>
      <td>64.56</td>
      <td>69.39</td>
      <td>75.65</td>
      <td>81.21</td>
      <td>59.33</td>
      <td>64.91</td>
      <td>64.22</td>
      <td>63.23</td>
      <td>68.93</td>
    </tr>
    <tr>
      <td>Base</td>
      <td>82.00</td>
      <td>60.57</td>
      <td>77.48</td>
      <td><strong>76.55</strong></td>
      <td>81.61</td>
      <td>62.03</td>
      <td>69.24</td>
      <td>57.85</td>
      <td>63.71</td>
      <td>68.16</td>
    </tr>
    <tr>
      <td>Visual-only PE(V)</td>
      <td>81.90</td>
      <td>60.57</td>
      <td>76.40</td>
      <td>66.21</td>
      <td><strong>83.82</strong></td>
      <td>60.30</td>
      <td>71.59</td>
      <td><strong>60.00</strong></td>
      <td>59.95</td>
      <td>69.66</td>
    </tr>
    <tr>
      <td>Audio-only PE(A)</td>
      <td><strong>82.09</strong></td>
      <td>59.40</td>
      <td><strong>78.48</strong></td>
      <td>76.22</td>
      <td>82.21</td>
      <td><strong>65.26</strong></td>
      <td><strong>74.65</strong></td>
      <td>58.48</td>
      <td><strong>67.60</strong></td>
      <td><strong>72.47</strong></td>
    </tr>
    <tr>
      <td>Audio+Visual PE(V+A)</td>
      <td>81.21</td>
      <td><strong>63.89</strong></td>
      <td>75.56</td>
      <td>71.42</td>
      <td>82.31</td>
      <td>60.30</td>
      <td>71.98</td>
      <td>57.94</td>
      <td>62.62</td>
      <td>70.27</td>
    </tr>
  </tbody>
</table>

