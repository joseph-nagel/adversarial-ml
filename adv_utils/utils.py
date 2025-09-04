'''Some utilities.'''

from urllib.request import urlretrieve


def download_file(url, save_path):
    '''Download and save file.'''
    save_path, _ = urlretrieve(url, filename=save_path)
    return save_path


def repeat_tensor(x, num_repeats, dim=0):
    '''Repeat tensor along axis.'''

    # create view
    if x.shape[dim] == 1:
        sizes = [-1] * x.ndim
        sizes[dim] = num_repeats
        out = x.expand(*sizes)

    # create copies
    else:
        repeats = [1] * x.ndim
        repeats[dim] = num_repeats
        out = x.repeat(*repeats)
        # out = x.tile(repeats)
        # out = x.repeat_interleave(repeats=num_repeats, dim=dim)

    return out
