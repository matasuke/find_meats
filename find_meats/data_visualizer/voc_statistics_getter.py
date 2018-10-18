from argparse import ArgumentParser
from typing import Union, List, Dict, Callable
import xml.etree.ElementTree as ET
from pathlib import Path
import pickle

from .base_statistics_getter import BaseStatisticsGetter, AnnotationInfo, ObjectsInfo
from .base_statistics_getter import BBOX_X_MIN, BBOX_X_MAX, BBOX_Y_MIN, BBOX_Y_MAX
from .base_statistics_getter import UNDEFINED_NAME, UNDEFINED_SHAPE, UNDEFINED_BBOX

VOC_FORMAT = '.xml'
IMG_FORMAT = '.jpg'
REG_EXP_IMG = f'*{IMG_FORMAT}'

class VocStatisticsGetter(BaseStatisticsGetter):

    def __init__(
            self,
            labels_text: List[str],
            filenames: List[str],
            label2filenames: Dict[str, List[str]],
            label2objects_num: Dict[str, int],
            label2images_num: Dict[str, int],
            image2objects: Dict[str, ObjectsInfo],
    ) -> None:
        super().__init__(
            labels_text,
            filenames,
            label2filenames,
            label2objects_num,
            label2images_num,
            image2objects,
        )

    @classmethod
    def create(
            cls,
            dataset_dir: Union[str, Path],
            process_annot_fn: Callable[[Union[str, Path]], AnnotationInfo]=None,
            annot_format: str=None,
    ) -> 'VocStatisticsGetter':
        '''
        generate statistics information from VOC format dataset.

        :param dataset_dir: dataset directory, which contains ANNOTATION_DIR and IMAGE_DIR as sub directory.
        :return: instance of this class.
        '''
        if process_annot_fn is None:
            process_annot_fn = cls.get_voc_info
        if annot_format is None:
            annot_format = VOC_FORMAT
        base_statistics_getter = super().create(dataset_dir, process_annot_fn, annot_format)

        return cls(
            base_statistics_getter._labels_text,
            base_statistics_getter._filenames,
            base_statistics_getter._label2filenames,
            base_statistics_getter._label2objects_num,
            base_statistics_getter._label2images_num,
            base_statistics_getter._image2objects,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'VocStatisticsGetter':
        '''
        Load saved voc format statistics information.

        :param path: path to save d statistical information.
        :return: instance of VocStatisticsGetter
        '''
        if isinstance(path, str):
            path = Path(path)
        assert path.exists()

        with open(path, 'rb') as f:
            (
                labels_text,
                filenames,
                label2filenames,
                label2objects_num,
                label2images_num,
                image2objects,
            ) = pickle.loads(f.read())

        return cls(
            labels_text,
            filenames,
            label2filenames,
            label2objects_num,
            label2images_num,
            image2objects,
        )

    def save(self, path: Union[str, Path]) -> None:
        '''
        load saved VocStatisticsGetter statistics information.

        :param save_path: path to saved statistical information.
        '''
        if isinstance(path, str):
            path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        with path.open('wb') as f:
            pickle.dump(
                (
                    self._labels_text,
                    self._filenames,
                    self._label2filenames,
                    self._label2objects_num,
                    self._label2images_num,
                    self._image2objects,
                ),
                f,
            )

    @classmethod
    def get_voc_info(
            cls,
            annotation_path: Union[str, Path],
    ) -> 'AnnotationInfo':
        '''
        get some information from xml file which is used for VOC format.

        :param annotation_path: annotation file path created as VOC format xml file.
        :return: extracted information.
        '''
        if isinstance(annotation_path, str):
            annotation_path = Path(annotation_path)
        assert annotation_path.exists()
        assert annotation_path.suffix == VOC_FORMAT

        tree = ET.parse(annotation_path.as_posix())
        root = tree.getroot()

        # File Name
        if root.find('filename') is None:
            file_name = 'NotDefined'
        else:
            file_name = root.find('filename').text  # type: ignore

        # Image shape
        if root.find('size') is None:
            shape = UNDEFINED_SHAPE
        else:
            img_size = root.find('size')
            shape = (
                int(img_size.find('height').text),  # type: ignore
                int(img_size.find('width').text),  # type: ignore
                int(img_size.find('depth').text),  # type: ignore
            )

        # Find annotations.
        bboxes = []
        labels_text = []
        difficult: List[int] = []
        truncated: List[int] = []

        for obj in root.findall('object'):

            # get name tag
            label = obj.find('name').text  # type:  ignore
            if label is None:
                label_name = UNDEFINED_NAME
            else:
                label_name = label.text  # type:  ignore
            labels_text.append(label_name)

            # get difficult tag
            difficult_tag = obj.find('difficult')
            if difficult_tag is None:
                difficult.append(0)
            else:
                difficult.append(int(difficult_tag.text))  # type:  ignore

            # get truncated tag
            truncated_tag = obj.find('truncated')
            if truncated is None:
                truncated.append(0)
            else:
                truncated.append(int(truncated_tag.text))  # type:  ignore

            # get bounding box tag
            bbox = obj.find('bndbox')
            if bbox is None:
                bboxes.append(UNDEFINED_BBOX)
            bboxes.append(
                {
                    BBOX_X_MIN: int(bbox.find(BBOX_X_MIN).text),  # type:  ignore
                    BBOX_X_MAX: int(bbox.find(BBOX_X_MAX).text),  # type:  ignore
                    BBOX_Y_MIN: int(bbox.find(BBOX_Y_MIN).text),  # type:  ignore
                    BBOX_Y_MAX: int(bbox.find(BBOX_Y_MAX).text),  # type:  ignore
                }
            )

        return AnnotationInfo(file_name, shape, bboxes, labels_text, difficult, truncated)


if __name__ == '__main__':
    parser = ArgumentParser(description='get statistical information about specified dataset')
    parser.add_argument('--dataset_dir', metavar='str', type=str, required=True)
    parser.add_argument()
    args = parser.parse_args()
