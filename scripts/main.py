'''
Training script.

Example
-------
python scripts/main.py fit --config config/adv_train.yaml

'''

from lightning.pytorch.cli import LightningCLI


def main():
    cli = LightningCLI()


if __name__ == '__main__':
    main()

