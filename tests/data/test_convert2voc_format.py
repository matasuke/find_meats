import os
from random import randint
from pathlib import Path
import pytest
import xml.etree.ElementTree as ET
from find_meats.data.convert2voc_format import (
    _process_annotation,
    _get_output_file_name,
    _train_test_split,
    _prepare_dirs,
    _convert,
    convert2voc_format,
)
from find_meats.data.convert2voc_format import (
    IMG_FORMAT,
    ANNOT_FORMAT,
    ANNOT_REG_EXP,
    IMG_REG_EXP,
    TARGET_ANNOT_DIR,
    TARGET_IMG_DIR,
    TRAIN_DIR,
)

TEST_DATASET_DIR = './tmp/data'
TEST_ANNOTATION_PATH = './tmp/data/labels/test_video_dir1/VID_20180619_212451/VID_20180619_212451_0.xml'
TEST_IMG_DIR = './tmp/data/processed'
TEST_ANNOT_DIR = './tmp/data/labels'
DATASET_NAME = 'MEAT_MASTER2018'
TEST_OUTPUT_DIR = './tmp/data/%s' % DATASET_NAME

app_env = os.environ.get('APP_ENV')

@pytest.mark.skipif(app_env=='CI', reason="CI environment doesn't have tmp dir")
def test_process_annotation(tmpdir):
    tmp_file = tmpdir.join('output.xml')
    tmp_file = Path(tmp_file)

    _process_annotation(TEST_ANNOTATION_PATH, tmp_file, DATASET_NAME)

    tree = ET.parse(tmp_file)
    root = tree.getroot()

    folder_name = root.find('folder').text
    file_name = root.find('filename').text
    path = root.find('path')

    assert folder_name == DATASET_NAME
    assert file_name == tmp_file.name
    assert path == None

def test_get_output_file_name():
    rand = randint(0, 100)
    EXPECTED = Path('%s/%05d%s' % (TEST_OUTPUT_DIR, rand, IMG_FORMAT))
    output_file_name = _get_output_file_name(TEST_OUTPUT_DIR, rand, IMG_FORMAT)

    assert output_file_name == EXPECTED

def test_train_test_split():
    data_list = list(range(100))
    SPLIT_NUM = 40
    EXPECTED_TRAIN = data_list[SPLIT_NUM:]
    EXPECTED_TEST = data_list[:SPLIT_NUM]

    list_train, list_test = _train_test_split(data_list, test_num=SPLIT_NUM, shuffle=False)

    assert list_train == EXPECTED_TRAIN
    assert list_test == EXPECTED_TEST

def test_train_test_split_with_shuffle():
    data_list = list(range(100))
    SPLIT_NUM = 40
    EXPECTED_TRAIN_NUM = 60
    EXPECTED_TEST_NUM = 40

    train_list, test_list = _train_test_split(data_list, test_num=SPLIT_NUM, shuffle=True)

    assert len(train_list) == EXPECTED_TRAIN_NUM
    assert len(test_list) == EXPECTED_TEST_NUM

@pytest.mark.skipif(app_env=='CI', reason="CI environment doesn't have tmp dir")
def test_convert(tmpdir):
    source_annot_dir = Path(TEST_ANNOT_DIR)
    source_img_dir = Path(TEST_IMG_DIR)
    source_annot_paths = [annot_path for annot_path in source_annot_dir.glob(ANNOT_REG_EXP)]
    tmp_annot_dir = tmpdir.mkdir('annotations')
    tmp_img_dir = tmpdir.mkdir('images')
    _convert(source_annot_paths, source_img_dir, tmp_annot_dir, tmp_img_dir, DATASET_NAME)

    for index in range(len(source_annot_paths)):
        target_annot_path = _get_output_file_name(tmp_annot_dir, index, ANNOT_FORMAT)
        target_img_path = _get_output_file_name(tmp_img_dir, index, IMG_FORMAT)

        assert target_annot_path.exists()
        assert target_img_path.exists()

@pytest.mark.skipif(app_env=='CI', reason="CI environment doesn't have tmp dir")
def test_prepare_dirs(tmpdir):
    EXPECTED_BASE_DIR = Path(tmpdir, TRAIN_DIR)
    EXPECTED_ANNOT_DIR = Path(EXPECTED_BASE_DIR, TARGET_ANNOT_DIR)
    EXPECTED_IMG_DIR = Path(EXPECTED_BASE_DIR, TARGET_IMG_DIR)

    target_aanot_dir, target_img_dir = _prepare_dirs(tmpdir, TRAIN_DIR)

    assert EXPECTED_ANNOT_DIR.exists()
    assert EXPECTED_IMG_DIR.exists()
    assert target_aanot_dir == EXPECTED_ANNOT_DIR
    assert target_img_dir == EXPECTED_IMG_DIR

@pytest.mark.skipif(app_env=='CI', reason="CI environment doesn't have tmp dir")
def test_convert2voc_format(tmpdir):
    source_annot_dir = Path(TEST_ANNOT_DIR)
    source_annot_paths_len = len([annot_path for annot_path in source_annot_dir.glob(ANNOT_REG_EXP)])
    tmp_annot_dir = Path(tmpdir, DATASET_NAME, TRAIN_DIR, TARGET_ANNOT_DIR)
    tmp_img_dir = Path(tmpdir, DATASET_NAME, TRAIN_DIR, TARGET_IMG_DIR)

    convert2voc_format(TEST_DATASET_DIR, tmpdir, DATASET_NAME)

    for index in range(source_annot_paths_len):
        target_annot_path = _get_output_file_name(tmp_annot_dir, index, ANNOT_FORMAT)
        target_img_path = _get_output_file_name(tmp_img_dir, index, IMG_FORMAT)

        assert target_annot_path.exists()
        assert target_img_path.exists()
