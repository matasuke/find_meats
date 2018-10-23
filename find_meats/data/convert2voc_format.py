from argparse import ArgumentParser
from pathlib import Path
import shutil
from tqdm import tqdm
from typing import Optional, Union, Tuple, List
import random
import xml.etree.ElementTree as ET

# path to source sub directories.
SOURCE_ANNOT_DIR = ['labels/']
SORUCE_IMG_DIR = ['processed/', 'pictures/']

# path to target sub directories, these are based on VOC2012 format.
TARGET_ANNOT_DIR = 'Annotations/'
TARGET_IMG_DIR = 'JPEGImages/'

# train, validation, test dir name
TRAIN_DIR = 'train'
VALID_DIR = 'valid'
TEST_DIR = 'test'

# image format to be allowed.
IMG_FORMAT = '.jpg'
ANNOT_FORMAT = '.xml'

# regular expression to get all annotations and images.
ANNOT_REG_EXP = '**/*%s' % ANNOT_FORMAT
IMG_REG_EXP = '**/*%s' % IMG_FORMAT

RANDOM_SEED = 4545

def _process_annotation(
        annotation_path: Union[str, Path],
        target_path: Union[str, Path],
        dataset_name: str,
) -> None:
    annotation_path = Path(annotation_path)
    target_path = Path(target_path)

    # prepare xml parser.
    tree = ET.parse(annotation_path.as_posix())
    root = tree.getroot()

    folder_name = root.find('folder')
    file_name = root.find('filename')
    path = root.find('path')
    if folder_name is not None:
        folder_name.text = dataset_name
    if file_name is not None:
        file_name.text = target_path.name
    if path is not None:
        root.remove(path)

    tree.write(target_path.as_posix())

def _get_output_file_name(
        output_dir: Union[str, Path],
        index: int,
        suffix: str,
) -> Path:
    output_dir = Path(output_dir)
    file_name = '%05d%s' % (index, suffix)
    return output_dir / file_name

def _train_test_split(
        data_list: List,
        test_num: int,
        shuffle: bool=True,
) -> Tuple:
    if shuffle:
        random.seed(RANDOM_SEED)
        random.shuffle(data_list)

    return data_list[test_num:], data_list[:test_num]

def _prepare_dirs(
        base_dir: Union[str, Path],
        suffix: str=None,
) -> Tuple:
    if not isinstance(base_dir, Path):
        base_dir = Path(base_dir)

    base_dir = base_dir / suffix if suffix else base_dir
    target_annot_dir = base_dir / TARGET_ANNOT_DIR
    target_img_dir = base_dir / TARGET_IMG_DIR

    if not base_dir.exists():
        base_dir.mkdir(parents=True)
    if not target_annot_dir.exists():
        target_annot_dir.mkdir()
    if not target_img_dir.exists():
        target_img_dir.mkdir()

    return target_annot_dir, target_img_dir

def _convert(
        source_annot_paths: Union[Path, List[Path]],
        source_img_dirs: Union[Path, List[Path]],
        target_annot_dir: Path,
        target_img_dir: Path,
        dataset_name: str,
) -> None:
    if not isinstance(source_annot_paths, list):
        source_annot_paths = [source_annot_paths]
    if not isinstance(source_img_dirs, list):
        source_img_dirs = [source_img_dirs]

    for index, annot_path in tqdm(enumerate(source_annot_paths)):
        target_annot_path = _get_output_file_name(target_annot_dir, index, ANNOT_FORMAT)
        target_img_path = _get_output_file_name(target_img_dir, index, IMG_FORMAT)

        source_img_name = '**/%s' % annot_path.with_suffix(IMG_FORMAT).name
        source_img_path = [
            img_path for source_dir in source_img_dirs for img_path in source_dir.glob(source_img_name)
        ] or None
        if source_img_path:
            shutil.copy(source_img_path[0], target_img_path)
            _process_annotation(annot_path, target_annot_path, dataset_name)

# TODO: apply exclusion file.
# TODO: add validation and test split. this has to consider data distribution.

def convert2voc_format(
        dataset_dir: str,
        output_dir: str,
        dataset_name: str='MEAT_MASTER2018',
        validation_size: Optional[int]=None,
        test_size: Optional[int]=None,
) -> None:
    '''
    convert original format created for MEAT2018 to VOC format.

    :param dataset_dir: path to original dataset directory which is converted to VOC format.
    :param output_dir: path to output directory.
    :param dataset_name: dataset name to be saved.
    :param validation_size: the size of validation dataset to be contained.
    :param test_size: the size of test dataset to be contained.
    '''
    source_base_dir = Path(dataset_dir)
    target_base_dir = Path(output_dir) / dataset_name

    assert source_base_dir.exists()
    assert validation_size is None or isinstance(validation_size, int)
    assert test_size is None or isinstance(test_size, int)

    target_annot_dir, target_img_dir = _prepare_dirs(target_base_dir, TRAIN_DIR)

    # get all original paths.
    source_annot_dirs = [source_base_dir / dir_name for dir_name in SOURCE_ANNOT_DIR]
    source_annot_paths = list(
        annot_path for source_dir in source_annot_dirs for annot_path in source_dir.glob(ANNOT_REG_EXP)
    )
    source_img_dirs = [source_base_dir / dir_name for dir_name in SORUCE_IMG_DIR]

    _convert(source_annot_paths, source_img_dirs, target_annot_dir, target_img_dir, dataset_name)


if __name__ == '__main__':
    parser = ArgumentParser(description='convert annotation and image files to voc format directories.')
    parser.add_argument('--dataset_name', metavar='str', type=str, default='MEAT_MASTER2018')
    parser.add_argument('--output_dir', metavar='str', type=str, required=True)
    parser.add_argument('--dataset_dir', metavar='str', type=str, required=True)
    parser.add_argument('--validation_size', metavar='INT', type=int, default=None)
    parser.add_argument('--test_size', metavar='INT', type=int, default=None)
    args = parser.parse_args()

    convert2voc_format(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        validation_size=args.validation_size,
        test_size=args.test_size,
    )
