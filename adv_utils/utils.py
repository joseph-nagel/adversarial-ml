'''Some utilities.'''

from urllib.request import urlretrieve


def download_file(url, save_path):
    '''Download and save file.'''
    save_path, _ = urlretrieve(url, filename=save_path)
    return save_path

