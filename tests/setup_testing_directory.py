import argparse
import os
from typing import Union
import warnings


def get_testing_directory() -> str:
    directory_file = 'testing_directory.txt'
    directory_files = [directory_file, os.path.join('tests', directory_file)]

    for directory_file in directory_files:
        if os.path.isfile(directory_file):
            with open(directory_file, 'r') as f:
                testing_directory = f.read()
                return testing_directory
    raise ValueError('please run setup_testing_directory.py before attempting to run unit tests')


def setup_testing_directory(datadir: Union[str, os.PathLike], overwrite: bool = False) -> str:
    testing_path_file = 'testing_directory.txt'

    should_setup = True
    if os.path.isfile(testing_path_file):
        with open(testing_path_file, 'r') as f:
            testing_directory = f.read()
            if not os.path.isfile(testing_directory):
                raise ValueError('saved testing directory {} does not exist, re-run ')
                warnings.warn(
                    'Saved testing directory {} does not exist, downloading Thumos14...'.format(testing_directory))
            else:
                should_setup = False
    if not should_setup:
        return testing_directory

    testing_directory = datadir
    assert os.path.isdir(testing_directory)
    assert os.path.isdir(os.path.join(testing_directory, 'train'))
    assert os.path.isdir(os.path.join(testing_directory, 'val'))
    with open('testing_directory.txt', 'w') as f:
        f.write(testing_directory)
    return testing_directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Setting up image directory for opencv transforms testing')
    parser.add_argument('-d', '--datadir', default=os.getcwd(), help='Imagenet directory')

    args = parser.parse_args()

    setup_testing_directory(args.datadir)