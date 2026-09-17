# CA1D

PyTorch implementation and pretrained encoder for the study:

**Lightweight Conventional Commits Classification with Multilingual Sentence Classification Pretraining**  
**다국어 문장 분류 사전학습을 활용한 경량 Conventional Commits 분류**

This repository contains the pretrained CA1D encoder and the notebook used for the Conventional Commits classification experiment.

## Overview

CA1D is a lightweight 1D convolution and attention-based text encoder.
Before Conventional Commits classification, the encoder was pretrained on **7 English sentence classification tasks and 3 Korean sentence classification tasks**.

For Conventional Commits classification, the model uses only:

- the masked commit message
- the **first line of the git diff**

The task consists of 10 classes:

`build`, `chore`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `style`, `test`

## Repository Contents

| File | Description |
|---|---|
| `encoder_pretrained.pth` | Pretrained CA1D encoder weights |
| `커밋메시지 분류.ipynb` | Model definition, tokenizer, fine-tuning, evaluation, and inference notebook |

## Model

The model uses a custom tokenizer with **256 token IDs**. English letters, digits, symbols, Hangul Jamo, frequent English/Korean strings, and selected emoji are represented in a compact shared token space. Unregistered Unicode characters are encoded using a special token followed by their hexadecimal code.

The encoder consists of:

- 64-dimensional token embedding
- three Window Attention stages
- stride-2 Conv1D downsampling between stages
- channel progression: `64 → 128 → 256 → 512`
- global self-attention blocks
- projection from 512 to 256 dimensions
- additional 256-dimensional self-attention blocks
- Rotary Position Embedding (RoPE)
- cross-attention pooling with 8 learnable queries

The full classification model contains approximately **17.42M parameters**, of which approximately **16.82M** belong to the encoder.

## Dataset

The Conventional Commits experiment uses `annotated_dataset.csv` released with:

> Qunhong Zeng, Yuxia Zhang, Zhiqing Qiu, and Hui Liu,  
> "A First Look at Conventional Commits Classification,"  
> Proceedings of the 47th IEEE/ACM International Conference on Software Engineering (ICSE), pp. 2277–2289, 2025.

Original repository:

https://github.com/0x404/conventional-commit-classification

The dataset contains **2,000 manually annotated commits**, with 200 samples for each of the 10 classes.

`annotated_dataset.csv` is not included in this repository. Place it in the same working directory as the notebook before running the experiment.

## Input Format

Each sample is constructed from the masked commit message and the first line of the git diff:

```text
{masked_commit_message}
-------
{first line of git_diff}
```

The first diff line usually contains information about the changed file path.

## Environment

The notebook uses:

```text
torch
pandas
joblib
scikit-learn
```

Install the dependencies with:

```bash
pip install torch pandas joblib scikit-learn
```

The uploaded notebook metadata records Python 3.13.2 and a GPU runtime. CUDA is used automatically when available.

## Running the Experiment

Clone the repository:

```bash
git clone https://github.com/mulnabi/CA1D.git
cd CA1D
```

Prepare the following files:

```text
CA1D/
├── encoder_pretrained.pth
├── annotated_dataset.csv
└── 커밋메시지 분류.ipynb
```

Then open `커밋메시지 분류.ipynb` in Jupyter Notebook or Google Colab and run the cells in order.

The notebook:

1. defines the CA1D architecture,
2. defines the custom tokenizer,
3. loads `encoder_pretrained.pth`,
4. reads `annotated_dataset.csv`,
5. creates stratified train/validation/test splits,
6. fine-tunes the classifier,
7. reports the confusion matrix and classification metrics,
8. performs a simple single-sentence inference example.

## Experimental Setup

| Split | Samples |
|---|---:|
| Train | 1,400 |
| Validation | 200 |
| Test | 400 |

The split is stratified by class and uses random seed `42`.

The experiment runs for 10 epochs with AdamW and cosine annealing. Evaluation uses exponential moving average (EMA) weights with decay `0.99`.

## Results

| Metric | Score |
|---|---:|
| Accuracy | **50.00%** |
| Macro Precision | **52.23%** |
| Macro Recall | **50.00%** |
| Macro F1 | **50.49%** |

CA1D uses only the commit message and the first line of the diff.

The test split used in this experiment is not identical to the split used for the baseline results reported by Zeng et al. Therefore, baseline comparisons should be interpreted descriptively rather than as paired statistical comparisons.

## Notes on Reproducibility

The repository provides the pretrained encoder and the notebook used for the experiment. The original annotated dataset must be obtained separately from the dataset authors' repository.

The pretrained encoder was trained on multilingual sentence classification tasks. The current study does not include an ablation experiment that isolates the causal effect of multilingual pretraining.

## Citation

If you use this repository, please cite:

```text
Sung-Yong Lim, "CA1D," GitHub Repository,
https://github.com/mulnabi/CA1D
```

For the Conventional Commits dataset, please also cite:

```text
Qunhong Zeng, Yuxia Zhang, Zhiqing Qiu, and Hui Liu,
"A First Look at Conventional Commits Classification,"
Proceedings of the 47th IEEE/ACM International Conference on Software Engineering (ICSE),
pp. 2277–2289, 2025.
```
