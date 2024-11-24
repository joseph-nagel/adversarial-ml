# Adversarial machine learning

This repository contains an exploration of adversarial attacks and defenses.


## Notebooks

- [Introduction](notebooks/intro.ipynb)

- [Adversarial attacks (PyTorch)](notebooks/adv_attacks_pt.ipynb)

- [Adversarial attacks (Hugging Face)](notebooks/adv_attacks_hf.ipynb)

- [Adversarial training](notebooks/adv_train.ipynb)


## Installation

```
pip install -e .
```


## Training

```
python scripts/main.py fit --config config/std_train.yaml
```

```
python scripts/main.py fit --config config/adv_train.yaml
```

