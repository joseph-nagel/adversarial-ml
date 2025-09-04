'''
Adversarial attacks.

Modules
-------
base : Attack base class.
fgsm : Fast-gradient sign method.
pgd : Projected gradient descent.

'''

from . import base, fgsm, pgd

from .base import AdversarialAttack

from .fgsm import fgsm_attack, FGSMAttack

from .pgd import pgd_attack, PGDAttack
