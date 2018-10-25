from pathlib import Path
from find_meats.data_visualizer.base_statistics_getter import ANNOTATION_DIR, IMAGE_DIR
from find_meats.data_visualizer.voc_statistics_getter import VOC_FORMAT, IMG_FORMAT
from find_meats.data_visualizer.base_statistics_getter import ObjectInfo
from find_meats.data_visualizer.base_statistics_getter import (
    BBOX_X_MIN, BBOX_X_MAX,
    BBOX_Y_MIN, BBOX_Y_MAX,
)

TEST_DATASET = Path('tmp', 'dataset')
ANNOTATION_DIR_PATH = Path(TEST_DATASET, ANNOTATION_DIR)
IMG_DIR_PATH = Path(TEST_DATASET, IMAGE_DIR)
SAVE_PATH = Path(TEST_DATASET, 'dataset.pkl')
ANNOTATION_PATH = Path(ANNOTATION_DIR_PATH, '00004.xml')

EXPECTED_LABELS_TEXT = [
    'raw_beaf',
    'cooked_beaf',
    'half_cooked_beaf',
    'raw_chicken',
    'half_cooked_chicken',
]

EXPECTED_FILENAMES = ['00000', '00001', '00002', '00003', '00004']

EXPECTED_LABEL2FILENAMES = {
    'raw_beaf': ['00000', '00003', '00004'],
    'half_cooked_beaf': ['00001'],
    'cooked_beaf': ['00001', '00003'],
    'raw_chicken': ['00002'],
    'half_cooked_chicken': ['00002'],
}

EXPECTED_LABEL2OBJECTS_NUM = {
    'raw_beaf': 7,
    'half_cooked_beaf': 2,
    'cooked_beaf': 5,
    'raw_chicken': 3,
    'half_cooked_chicken': 2,
}

EXPECTED_OBJECTINFO_00000 = [
    ObjectInfo(
        label_text='raw_beaf',
        bbox={
            BBOX_X_MIN: 216,
            BBOX_X_MAX: 395,
            BBOX_Y_MIN: 507,
            BBOX_Y_MAX: 639,
        },
        difficult=False,
        truncated=False,
    ),
    ObjectInfo(
        label_text='raw_beaf',
        bbox={
            BBOX_X_MIN: 401,
            BBOX_X_MAX: 582,
            BBOX_Y_MIN: 627,
            BBOX_Y_MAX: 751,
        },
        difficult=False,
        truncated=False
    ),
    ObjectInfo(
        label_text='raw_beaf',
        bbox={
            BBOX_X_MIN: 73,
            BBOX_X_MAX: 301,
            BBOX_Y_MIN: 597,
            BBOX_Y_MAX: 744,
        },
        difficult=False,
        truncated=False,
    )
]

EXPECTED_OBJECTINFO_00001 = [
    ObjectInfo(
        label_text='cooked_beaf',
        bbox={
            BBOX_X_MIN: 300,
            BBOX_X_MAX: 458,
            BBOX_Y_MIN: 498,
            BBOX_Y_MAX: 624,
        },
        difficult=False,
        truncated=False,
    ),
    ObjectInfo(
        label_text='cooked_beaf',
        bbox={
            BBOX_X_MIN: 481,
            BBOX_X_MAX: 633,
            BBOX_Y_MIN: 620,
            BBOX_Y_MAX: 748,
        },
        difficult=False,
        truncated=False,
    ),
    ObjectInfo(
        label_text='cooked_beaf',
        bbox={
            BBOX_X_MIN: 364,
            BBOX_X_MAX: 594,
            BBOX_Y_MIN: 655,
            BBOX_Y_MAX: 833,
        },
        difficult=False,
        truncated=False,
    ),
    ObjectInfo(
        label_text='half_cooked_beaf',
        bbox={
            BBOX_X_MIN: 191,
            BBOX_X_MAX: 394,
            BBOX_Y_MIN: 691,
            BBOX_Y_MAX: 852,
        },
        difficult=False,
        truncated=False,
    ),
    ObjectInfo(
        label_text='half_cooked_beaf',
        bbox={
            BBOX_X_MIN: 136,
            BBOX_X_MAX: 372,
            BBOX_Y_MIN: 599,
            BBOX_Y_MAX: 704,
        },
        difficult=False,
        truncated=False,
    ),
]

EXPECTED_OBJECTINFO_00002=[
    ObjectInfo(
        label_text='half_cooked_chicken',
        bbox={
            BBOX_X_MIN: 417,
            BBOX_X_MAX: 603,
            BBOX_Y_MIN: 411,
            BBOX_Y_MAX: 563,
        },
        difficult=False,
        truncated=False,
    ),
    ObjectInfo(
        label_text='half_cooked_chicken',
        bbox={
            BBOX_X_MIN: 403,
            BBOX_X_MAX: 628,
            BBOX_Y_MIN: 552,
            BBOX_Y_MAX: 719,
        },
        difficult=False,
        truncated=False,
    ),
    ObjectInfo(
        label_text='raw_chicken',
        bbox={
            BBOX_X_MIN: 336,
            BBOX_X_MAX: 592,
            BBOX_Y_MIN: 716,
            BBOX_Y_MAX: 924,
        },
        difficult=False,
        truncated=False,
    ),
    ObjectInfo(
        label_text='raw_chicken',
        bbox={
            BBOX_X_MIN: 217,
            BBOX_X_MAX: 451,
            BBOX_Y_MIN: 610,
            BBOX_Y_MAX: 782,
        },
        difficult=False,
        truncated=False,
    ),
    ObjectInfo(
        label_text='raw_chicken',
        bbox={
            BBOX_X_MIN: 33,
            BBOX_X_MAX: 329,
            BBOX_Y_MIN: 723,
            BBOX_Y_MAX: 904,
        },
        difficult=False,
        truncated=False,
    ),
]

EXPECTED_OBJECTINFO_00003 = [
    ObjectInfo(
        label_text='cooked_beaf',
        bbox={
            BBOX_X_MIN: 457,
            BBOX_X_MAX: 606,
            BBOX_Y_MIN: 624,
            BBOX_Y_MAX: 747,
        },
        difficult=False,
        truncated=False,
    ),
    ObjectInfo(
        label_text='cooked_beaf',
        bbox={
            BBOX_X_MIN: 272,
            BBOX_X_MAX: 429,
            BBOX_Y_MIN: 499,
            BBOX_Y_MAX: 620,
        },
        difficult=False,
        truncated=False,
    ),
    ObjectInfo(
        label_text='raw_beaf',
        bbox={
            BBOX_X_MIN: 367,
            BBOX_X_MAX: 597,
            BBOX_Y_MIN: 691,
            BBOX_Y_MAX: 845,
        },
        difficult=False,
        truncated=False,
    ),
    ObjectInfo(
        label_text='raw_beaf',
        bbox={
            BBOX_X_MIN: 183,
            BBOX_X_MAX: 381,
            BBOX_Y_MIN: 717,
            BBOX_Y_MAX: 882,
        },
        difficult=False,
        truncated=False,
    ),
    ObjectInfo(
        label_text='raw_beaf',
        bbox={
            BBOX_X_MIN: 117,
            BBOX_X_MAX: 331,
            BBOX_Y_MIN: 580,
            BBOX_Y_MAX: 702,
        },
        difficult=False,
        truncated=False,
    ),
]

EXPECTED_OBJECTINFO_00004 = [
    ObjectInfo(
        label_text='raw_beaf',
        bbox={
            BBOX_X_MIN: 261,
            BBOX_Y_MIN: 517,
            BBOX_X_MAX: 431,
            BBOX_Y_MAX: 644,
        },
        difficult=False,
        truncated=False,
    ),
]

EXPECTED_FILENAME2OBJECTS = {
    '00000': EXPECTED_OBJECTINFO_00000,
    '00001': EXPECTED_OBJECTINFO_00001,
    '00002': EXPECTED_OBJECTINFO_00002,
    '00003': EXPECTED_OBJECTINFO_00003,
    '00004': EXPECTED_OBJECTINFO_00004,
}

EXPECTED_FILENAME2SHAPE = {
    '00000': (720, 1280, 3),
    '00001': (720, 1280, 3),
    '00002': (720, 1280, 3),
    '00003': (720, 1280, 3),
    '00004': (720, 1280, 3),
}

EXPECTED_FILENAME2ANNOT_PATH = {
    '00000': str(ANNOTATION_DIR_PATH / f'00000{VOC_FORMAT}'),
    '00001': str(ANNOTATION_DIR_PATH / f'00001{VOC_FORMAT}'),
    '00002': str(ANNOTATION_DIR_PATH / f'00002{VOC_FORMAT}'),
    '00003': str(ANNOTATION_DIR_PATH / f'00003{VOC_FORMAT}'),
    '00004': str(ANNOTATION_DIR_PATH / f'00004{VOC_FORMAT}'),
}

EXPECTED_FILENAME2IMG_PATH = {
    '00000': str(IMG_DIR_PATH / f'00000{IMG_FORMAT}'),
    '00001': str(IMG_DIR_PATH / f'00001{IMG_FORMAT}'),
    '00002': str(IMG_DIR_PATH / f'00002{IMG_FORMAT}'),
    '00003': str(IMG_DIR_PATH / f'00003{IMG_FORMAT}'),
    '00004': str(IMG_DIR_PATH / f'00004{IMG_FORMAT}'),
}

EXPECTED_LABEL2IMAGES_NUM = {
    'raw_beaf': 3,
    'half_cooked_beaf': 1,
    'cooked_beaf': 2,
    'raw_chicken': 1,
    'half_cooked_chicken': 1,
}

EXPECTED_FILENAME2LABELS = {
    '00000': ['raw_beaf', 'raw_beaf', 'raw_beaf'],
    '00001': ['cooled_beaf', 'cooled_beaf', 'cooled_beaf',
              'half_cooked_beaf', 'half_cooked_beaf'
              ],
    '00002': ['half_cooked_chicken', 'half_cooked_chicken',
              'raw_chicken', 'raw_chicken', 'raw_chicken'],
    '00003': ['raw_beaf', 'raw_beaf', 'raw_beaf',
              'cooked_beaf', 'cooked_beaf'
              ],
    '00004': ['raw_beaf'],
}
