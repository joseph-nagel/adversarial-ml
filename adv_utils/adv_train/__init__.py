'''
Adversarial training.

Modules
-------
base : Adv. training base class.
hugging : Adv. training for Hugging Face models.

'''

from . import base, hugging

from .base import AdversarialTraining

from .hugging import AdversarialHFClassifier

