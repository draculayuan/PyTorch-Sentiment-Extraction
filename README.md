# PyTorch-Sentiment-Extraction

A code base for Kaggle's Tweet Sentiment Extraction Competition based on PyTorch

## Author
- [Liu Yuan](https://github.com/draculayuan)


## Get Started

1. Install dependencies:
    - [pytorch>=0.4](https://pytorch.org/)
    - [transformers>=2.8]( by hugging face)


## Train

```bash
sh scripts/train.sh path/to/your/configfile model_name
```


## Test

```bash
sh scripts/test.sh path/to/your/configfile path/to/saved/model
```

## Inference

```bash
sh scripts/infer.sh path/to/your/configfile path/to/saved/model path/to/output/path
```

