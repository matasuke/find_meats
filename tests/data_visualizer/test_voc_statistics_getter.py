import pytest
from pathlib import Path
import pytest
from find_meats.data_visualizer.voc_statistics_getter import VocStatisticsGetter
from find_meats.data_visualizer.voc_statistics_getter import IMG_FORMAT, VOC_FORMAT
from find_meats.data_visualizer.base_statistics_getter import (
    BBOX_X_MIN,
    BBOX_X_MAX,
    BBOX_Y_MIN,
    BBOX_Y_MAX,
)
from .test_configurations import (
    TEST_DATASET,
    ANNOTATION_DIR_PATH,
    IMG_DIR_PATH,
    SAVE_PATH,
    ANNOTATION_PATH,
    EXPECTED_LABELS_TEXT,
    EXPECTED_FILENAMES,
    EXPECTED_LABEL2FILENAMES,
    EXPECTED_LABEL2IMAGES_NUM,
    EXPECTED_LABEL2OBJECTS_NUM,
    EXPECTED_FILENAME2OBJECTS,
    EXPECTED_FILENAME2SHAPE,
    EXPECTED_FILENAME2ANNOT_PATH,
    EXPECTED_FILENAME2IMG_PATH,
    EXPECTED_FILENAME2LABELS,
)


def test_create():
    voc_statistics_getter = VocStatisticsGetter.create(TEST_DATASET)
    labels_text = voc_statistics_getter._labels_text
    filenames = voc_statistics_getter._filenames
    # test labels
    for label_text in labels_text:
        assert label_text in EXPECTED_LABELS_TEXT

    # test filenames
    assert len(filenames) == len(EXPECTED_FILENAMES)
    for file_name in filenames:
        assert file_name in EXPECTED_FILENAMES

    # test label2filenames
    assert len(voc_statistics_getter._label2filenames) == len(EXPECTED_LABELS_TEXT)
    for label_text in labels_text:
        filenames_list = voc_statistics_getter._label2filenames[label_text]
        expected_filenames_list = EXPECTED_LABEL2FILENAMES[label_text]

        assert len(filenames_list) == len(expected_filenames_list)
        for file_name in filenames_list:
            assert file_name in expected_filenames_list

    # test label2objects_num
    assert len(voc_statistics_getter._label2objects_num) == len(EXPECTED_LABELS_TEXT)
    for label_text in labels_text:
        objects_list = voc_statistics_getter._label2objects_num[label_text]
        assert objects_list == EXPECTED_LABEL2OBJECTS_NUM[label_text]

    # test filename2objects
    for file_name in filenames:
        objects = voc_statistics_getter._filename2objects[file_name]
        expected_objects = EXPECTED_FILENAME2OBJECTS[file_name]
        assert len(objects) == len(expected_objects)
        for obj, expected in zip(objects, expected_objects):
            assert obj.label_text == expected.label_text
            assert obj.bbox == expected.bbox
            assert obj.difficult == expected.difficult
            assert obj.truncated == expected.truncated

    # test filename2shape
    for file_name in filenames:
        img_shape = voc_statistics_getter._filename2shape[file_name]
        expected_shape = EXPECTED_FILENAME2SHAPE[file_name]
        assert img_shape == expected_shape

    # test filename2anot_path
    for file_name in filenames:
        annot_path = voc_statistics_getter._filename2annot_path[file_name]
        expected_annot_path = EXPECTED_FILENAME2ANNOT_PATH[file_name]
        assert annot_path == expected_annot_path

    # test filename2img_path
    for file_name in filenames:
        img_path = voc_statistics_getter._filename2img_path[file_name]
        expected_img_path = EXPECTED_FILENAME2IMG_PATH[file_name]
        assert img_path == expected_img_path

def test_save_and_load(tmpdir):
    voc_statistics_getter = VocStatisticsGetter.create(TEST_DATASET)
    voc_statistics_getter.save(SAVE_PATH)
    loaded_data = VocStatisticsGetter.load(SAVE_PATH)

    assert len(voc_statistics_getter._labels_text) == len(loaded_data._labels_text)
    assert len(voc_statistics_getter._filenames) == len(loaded_data._filenames)
    assert len(voc_statistics_getter._label2filenames) == len(loaded_data._filenames)
    assert len(voc_statistics_getter._label2objects_num) == len(loaded_data._label2objects_num)
    assert len(voc_statistics_getter._filename2objects) == len(loaded_data._filename2objects)
    assert len(voc_statistics_getter._filename2shape) == len(loaded_data._filename2shape)
    assert len(voc_statistics_getter._filename2annot_path) == len(loaded_data._filename2annot_path)
    assert len(voc_statistics_getter._filename2img_path) == len(loaded_data._filename2img_path)


def test_get_voc_info():
    EXPECTED_FILENAME = '00004'
    EXPECTED_SHAPE = (720, 1280, 3)
    EXPECTED_BBOX = [{BBOX_X_MIN: 261, BBOX_X_MAX: 431, BBOX_Y_MIN: 517, BBOX_Y_MAX: 644}]
    EXPECTED_LABEL = ['raw_beaf']
    EXPECTED_DIFFICULT = [False]
    EXPECTED_TRUNCATED = [False]

    annot_info = VocStatisticsGetter.get_voc_info(ANNOTATION_PATH)

    assert annot_info.filename == EXPECTED_FILENAME
    assert annot_info.shape == EXPECTED_SHAPE
    assert annot_info.bboxes == EXPECTED_BBOX
    assert annot_info.labels_text == EXPECTED_LABEL
    assert annot_info.difficult == EXPECTED_DIFFICULT
    assert annot_info.truncated == EXPECTED_TRUNCATED

def test_image_num(loaded_voc_getter):
    EXPECTED_IMG_NUM = 5

    assert loaded_voc_getter.images_num == EXPECTED_IMG_NUM

def test_objects_num(loaded_voc_getter):
    EXPECTED_OBJECTS_NUM = 19

    assert loaded_voc_getter.objects_num == EXPECTED_OBJECTS_NUM

def test_difficult_num(loaded_voc_getter):
    EXPECTED_DIFFICULT_NUM = 0

    assert loaded_voc_getter.difficult_num == EXPECTED_DIFFICULT_NUM

def test_truncated_num(loaded_voc_getter):
    EXPECTED_TRUNCATED_NUM = 0

    assert loaded_voc_getter.truncated_num == EXPECTED_TRUNCATED_NUM

def test_labels(loaded_voc_getter):

    assert len(loaded_voc_getter.labels) == len(EXPECTED_LABELS_TEXT)
    for label in loaded_voc_getter.labels:
        assert label in EXPECTED_LABELS_TEXT

def test_filenames(loaded_voc_getter):

    assert len(loaded_voc_getter.labels) == len(EXPECTED_LABELS_TEXT)
    for label in loaded_voc_getter.labels:
        assert label in EXPECTED_LABELS_TEXT

def test_annot2img(loaded_voc_getter):
    FILENAME = '00000'

    annot_path = loaded_voc_getter._filename2annot_path[FILENAME]
    img_path = loaded_voc_getter._filename2img_path[FILENAME]

    converted = loaded_voc_getter._annot2img(annot_path, IMG_FORMAT)
    assert converted == img_path

def test_img2annot(loaded_voc_getter):
    FILENAME = '00000'

    img_path = loaded_voc_getter._filename2img_path[FILENAME]
    annot_path = loaded_voc_getter._filename2annot_path[FILENAME]

    converted = loaded_voc_getter._img2annot(img_path, VOC_FORMAT)
    assert converted == annot_path

def test_label2images_num(loaded_voc_getter):
    for label in EXPECTED_LABELS_TEXT:
        img_num = loaded_voc_getter.label2images_num(label)
        expected = EXPECTED_LABEL2IMAGES_NUM[label]
        assert img_num == expected

def test_label2objects_num(loaded_voc_getter):
    for label in EXPECTED_LABELS_TEXT:
        obj_num = loaded_voc_getter.label2objects_num(label)
        expected = EXPECTED_LABEL2OBJECTS_NUM[label]
        assert obj_num == expected

def test_label2filenames(loaded_voc_getter):
    for label in EXPECTED_LABELS_TEXT:
        labels = loaded_voc_getter.label2filenames(label)
        expected = EXPECTED_LABEL2FILENAMES[label]
        assert len(labels) == len(expected)

def test_filename2image_path(loaded_voc_getter):
    for filename in EXPECTED_FILENAMES:
        image_path = loaded_voc_getter.filename2image_path(filename)
        expected = EXPECTED_FILENAME2IMG_PATH[filename]
        assert image_path == expected

def test_filename2annot_path(loaded_voc_getter):
    for filename in EXPECTED_FILENAMES:
        annot_path = loaded_voc_getter.filename2annotation_path(filename)
        expected = EXPECTED_FILENAME2ANNOT_PATH[filename]
        assert annot_path == expected

def test_filename2objects(loaded_voc_getter):
    for filename in EXPECTED_FILENAMES:
        objs = loaded_voc_getter.filename2objects(filename)
        expected = EXPECTED_FILENAME2OBJECTS[filename]
        assert len(objs) == len(expected)

def test_filename2shape(loaded_voc_getter):
    for filename in EXPECTED_FILENAMES:
        img_shape = loaded_voc_getter.filename2shape(filename)
        expected = EXPECTED_FILENAME2SHAPE[filename]
        assert img_shape == expected

def test_filename2labels(loaded_voc_getter):
    for filename in EXPECTED_FILENAMES:
        labels = loaded_voc_getter.filename2labels(filename)
        expected = EXPECTED_FILENAME2LABELS[filename]
        assert len(labels) == len(expected)

def test_filename2bboxes(loaded_voc_getter):
    for filename in EXPECTED_FILENAMES:
        bboxes = loaded_voc_getter.filename2bboxes(filename)
        expected = []

        objects_info = loaded_voc_getter._filename2objects[filename]
        for obj_info in objects_info:
            expected.append(obj_info.bbox)

        assert len(bboxes) == len(expected)
        for bbox in bboxes:
            assert bbox in expected

def test_filename2difficult(loaded_voc_getter):
    for filename in EXPECTED_FILENAMES:
        difficult = loaded_voc_getter.filename2difficult(filename)
        expected = []

        objects_info = loaded_voc_getter._filename2objects[filename]
        for obj_info in objects_info:
            expected.append(obj_info.difficult)

        assert len(difficult) == len(expected)
        for diff in difficult:
            assert diff in expected

def test_filename2truncated(loaded_voc_getter):
    for filename in EXPECTED_FILENAMES:
        truncated = loaded_voc_getter.filename2truncated(filename)
        expected = []

        objects_info = loaded_voc_getter._filename2objects[filename]
        for obj_info in objects_info:
            expected.append(obj_info.truncated)

        assert len(truncated) == len(expected)
        for trunc in truncated:
            assert trunc in expected

def test_absolute2relative_bbox_position(loaded_voc_getter):
    IMG_SHAPE = (720, 1200, 3)
    BBOXES = [
        {
            BBOX_X_MIN: 183,
            BBOX_X_MAX: 381,
            BBOX_Y_MIN: 717,
            BBOX_Y_MAX: 882,
        }
    ]
    EXPECTED_BBOXES = [
        {
            BBOX_X_MIN: 0.254,
            BBOX_X_MAX: 0.529,
            BBOX_Y_MIN: 0.598,
            BBOX_Y_MAX: 0.735,
        }
    ]

    relative_bboxes = loaded_voc_getter.absolute2relative_bbox_position(BBOXES, IMG_SHAPE)
    assert relative_bboxes == EXPECTED_BBOXES
