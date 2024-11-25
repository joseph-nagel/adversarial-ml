# Adversarial machine learning

This repository contains an exploration of adversarial attacks and defenses.


<p>
  <img src="assets/original.png" alt="The original image is correctly classified as a volcano" title="Original image before the attack" height="250" style="padding-right: 1em;">
  <img src="assets/attacked.png" alt="The attacked image is misclassified as a goldfish" title="Perturbed image after the (targeted PGD) attack" height="250">
</p>


## Notebooks

- [Introduction](notebooks/intro.ipynb)

- [Adversarial attacks (ART)](notebooks/adv_attacks_art.ipynb)

- [Adversarial attacks (PyTorch)](notebooks/adv_attacks_pt.ipynb)

- [Adversarial attacks (Hugging Face)](notebooks/adv_attacks_hf.ipynb)

- [Adversarial training (Hugging Face)](notebooks/adv_train_hf.ipynb)


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

