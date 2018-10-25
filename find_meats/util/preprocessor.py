import random
from typing import List, Tuple

RANDOM_SEED = 4545

def train_test_split(
        data_list: List,
        test_num: int = None,
        test_ratio: float = None,
        shuffle: bool = True,
) -> Tuple:
    '''
    split data into train and test.

    :param data_list: list to split.
    :param test_num: the number of data to be contained in test data.
    :param test_ratio: ratio to be contained in test data.
    :param shuffle: shuffle data_list.
    '''
    assert (test_num is None and test_ratio is not None) or \
        (test_num is not None and test_ratio is None)
    assert test_num is None or test_num < len(data_list)
    assert test_ratio is None or test_ratio < 1.0

    if shuffle:
        random.seed(RANDOM_SEED)
        random.shuffle(data_list)

    if test_ratio is not None:
        test_num = int(test_ratio * len(data_list))

    if test_num is not None:
        split_point = len(data_list) - test_num

    return data_list[:split_point], data_list[split_point:]
