# Adversarial machine learning

This repository contains an exploration of adversarial attacks and defenses.


## Notebooks

- [Introduction](notebooks/intro.ipynb)

- [Adversarial attacks (PyTorch)](notebooks/attacks_pt.ipynb)

- [Adversarial attacks (Hugging Face)](notebooks/attacks_hf.ipynb)


## Installation

```
pip install -e .
```


## Training

```
python scripts/main.py fit --config config/train_std.yaml
```

```
python scripts/main.py fit --config config/train_adv.yaml
```

