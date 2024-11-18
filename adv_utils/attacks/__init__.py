'''Adversarial attacks.'''

from . import fgsm, pgd

from .fgsm import fgsm_attack, FGSMAttack

from .pgd import pgd_attack, PGDAttack

