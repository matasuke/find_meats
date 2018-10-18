from pathlib import Path
from typing import Callable, Union, List, Mapping, Tuple, NamedTuple, Dict
from tqdm import tqdm

ANNOTATION_DIR = 'Annotations'
IMAGE_DIR = 'JPEGImages'

# bounding box info.
BBOX_X_MIN = 'xmin'
BBOX_X_MAX = 'xmax'
BBOX_Y_MIN = 'ymin'
BBOX_Y_MAX = 'ymax'

# replacement of undefined name
UNDEFINED_NAME = 'UNDEFINED'
UNDEFINED_SHAPE = (-1, -1, -1)
UNDEFINED_BBOX = {BBOX_X_MIN: -1, BBOX_X_MAX: -1, BBOX_Y_MIN: -1, BBOX_Y_MAX: -1}

class BaseStatisticsGetter:
    __slots__ = [
        '_labels_text',
        '_filenames',
        '_label2filenames',
        '_label2objects_num',
        '_label2images_num',
        '_image2objects',
    ]

    def __init__(
            self,
            labels_text: List[str],
            filenames: List[str],
            label2filenames: Dict[str, List[str]],
            label2objects_num: Dict[str, int],
            label2images_num: Dict[str, int],
            image2objects: Dict[str, 'ObjectsInfo'],
    ) -> None:
        self._labels_text = labels_text
        self._filenames = filenames
        self._label2filenames = label2filenames
        self._label2objects_num = label2objects_num
        self._label2images_num = label2images_num
        self._image2objects = image2objects

    @property
    def images_num(self) -> int:
        '''
        get the total number of images.
        '''
        total_img_num = 0
        for img_num in self._label2images_num.values():
            total_img_num += img_num

        return total_img_num

    @property
    def objects_num(self) -> int:
        '''
        get the total number of objects.
        '''
        total_obj_num = 0
        for obj_num in self._label2objects_num.values():
            total_obj_num += obj_num

        return total_obj_num

    @property
    def difficult_num(self) -> int:
        '''
        get the total number of difficult.
        '''
        total_difficult = 0
        for filename in self._filenames:
            total_difficult += sum(self.image2difficult(filename))

        return total_difficult

    @property
    def truncated_num(self) -> int:
        '''
        get the total number of truncated.
        '''
        total_truncated = 0
        for filename in self._filenames:
            total_truncated += sum(self.image2truncated(filename))

        return total_truncated

    @property
    def labels(self) -> List[str]:
        '''
        get the list of label names.
        '''
        return self._labels_text

    @property
    def filenames(self) -> List[str]:
        '''
        get the list of filenames.
        '''
        return self._filenames

    @classmethod
    def create(
            cls,
            dataset_dir: Union[str, Path],
            process_annot_fn: Callable[[Union[str, Path]], 'AnnotationInfo'],
            annot_format: str,
    ) -> 'BaseStatisticsGetter':
        '''
        create BaseStatisticsGetter.

        :param dataset_dir: dataset directory, which contains ANNOTATION_DIR and IMAGE_DIR as sub directory.
        :param process_annot_fn: processor for some format of annotation.
        :param annot_format: format of annotation file.
        :return : instance of this class.
        '''
        if isinstance(dataset_dir, str):
            dataset_dir = Path(dataset_dir)

        annot_dir = dataset_dir / ANNOTATION_DIR
        img_dir = dataset_dir / IMAGE_DIR
        assert dataset_dir.exists()
        assert annot_dir.exists()
        assert img_dir.exists()

        reg_exp_annot = f'*{annot_format}'
        annot_files = annot_dir.glob(reg_exp_annot)

        labels_text: List[str] = []
        filenames: List[str] = []
        label2filenames: Dict[str, List[str]] = {}
        label2objects_num: Dict[str, int] = {}
        label2images_num: Dict[str, int] = {}
        image2objects: Dict[str, 'ObjectsInfo'] = {}

        for annot in tqdm(annot_files):
            annot_info = process_annot_fn(annot)
            filename = annot.stem
            filenames.append(filename)

            # count the number of objects tagged by label.
            for label_text in annot_info.labels_text:
                if label_text in label2objects_num:
                    label2objects_num[label_text] += 1
                else:
                    label2objects_num[label_text] = 1

            # count the number of images taggeed by label.
            for label_text in set(annot_info.labels_text):
                label2filenames[label_text].append(filename)
                if label_text in label2images_num:
                    label2images_num[label_text] += 1
                else:
                    label2images_num[label_text] = 1

            # create statistics information about bboxes.
            bboxes = annot_info.bboxes
            difficult = annot_info.difficult
            truncated = annot_info.truncated
            objects_info = ObjectsInfo(bboxes, difficult, truncated)
            image2objects[filename] = objects_info

        labels_text = list(label2objects_num.keys())
        return cls(
            labels_text,
            filenames,
            label2filenames,
            label2objects_num,
            label2images_num,
            image2objects,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseStatisticsGetter':
        '''
        load saved statistics information.

        :param path: path to saved statistical information.
        '''
        raise NotImplementedError()

    def save(self, path: Union[str, Path]) -> None:
        '''
        save created statistical information.

        :param path: pah to save directory.
        '''
        raise NotImplementedError()

    def label2images_num(self, label: str) -> int:
        '''
        get the number of images tagged by label.

        :param label: label name to get the number of images.
        :return: the number of images.
        '''
        return self._label2images_num[label]

    def label2objects_num(self, label: str) -> int:
        '''
        get the number of objects tagged by label.

        :param label: label name to get the number of images.
        :return: the number of images.
        '''
        return self._label2objects_num[label]

    def label2filenames(self, label: str,) -> List[str]:
        '''
        get filenames tagged by label.

        :param label: label text to get the list of file names.
        :return: list of filenames tagged by label.
        '''
        return self._label2filenames[label]

    def _img2annot(
            self,
            img_path: Union[str, Path],
            annot_format: str,
    ) -> Path:
        '''
        get annotation path from image path.
        image path has to be <parent of parent dir>/<ANNOTATION_DIR>/<annotation path>

        :param img_path: image path.
        :param annot_format: format of annotation.
        :return: annotation path.
        '''
        if isinstance(img_path, str):
            img_path = Path(img_path)
        assert img_path.exists()

        annot_file = img_path.with_suffix(annot_format).name
        annot_path = img_path.parents[1] / ANNOTATION_DIR / annot_file

        return annot_path

    def _annot2img(
            self,
            annot_path: Union[Path, str],
            img_format: str,
    ) -> Path:
        '''
        get image path from annotation path.
        image path has to be <parent of parent dir>/<IAMGE_DIR>/<image path>

        :param annot_path: annotation path.
        :param img_format: format of image.
        :return image path.
        '''
        if isinstance(annot_path, str):
            annot_path = Path(annot_path)
        assert annot_path.exists()

        img_file = annot_path.with_suffix(img_format).name
        img_path = annot_path.parents[1] / IMAGE_DIR / img_file

        return img_path

    def image2objects(self, filename: str) -> List[Dict[str, int]]:
        '''
        get the labels specified in an image.

        :param filename: filename for getting object information.
        :return: dict of label_text and bboxes.
        '''
        objects_info = self._image2objects[filename]
        return objects_info.bboxes

    def image2bboxes(self, filename: str) -> List[Dict[str, int]]:
        '''
        get bboxes specified in the image.

        :param filename: filename for getting bboxes.
        :return: list of bboxes in the image.
        '''
        objects_info = self._image2objects[filename]
        return objects_info.bboxes

    def image2difficult(self, filename: str) -> List[bool]:
        '''
        get difficult specified in the image.

        :param filename: filename for getting difficult.
        :return: list of difficult in the image.
        '''
        objects_info = self._image2objects[filename]
        difficult = list(map(bool, objects_info.difficult))

        return difficult

    def image2truncated(self, filename: str) -> List[bool]:
        '''
        get truncated specified in the image.

        :param filename: filename for getting truncated.
        :return: list of truncated in the image.
        '''
        objects_info = self._image2objects[filename]
        truncated = list(map(bool, objects_info.truncated))

        return truncated

    def absolute2relative_bbox_position(
            self,
            img_shape: Tuple[int, int, int],
            bboxes: List[Mapping[str, int]],
    ) -> List[Dict[str, float]]:
        '''
        convert absolute position of bbox into relative position.

        :param img_shape: image shape of target image.
        :param bboxes: list of bounding boxes.
        :return: converted bounding boxes, which are relative position in an image.
        '''
        img_height, img_width, img_channel = img_shape

        relative_bboxes = []
        for bbox in bboxes:
            relative_bboxes.append(
                {
                    BBOX_X_MIN: float(bbox[BBOX_X_MIN]) / img_width,
                    BBOX_X_MAX: float(bbox[BBOX_X_MAX]) / img_width,
                    BBOX_Y_MIN: float(bbox[BBOX_Y_MIN]) / img_height,
                    BBOX_Y_MAX: float(bbox[BBOX_Y_MAX]) / img_height,
                }
            )

        return relative_bboxes

class AnnotationInfo(NamedTuple):
    filename: str
    shape: Tuple[int, int, int]
    bboxes: List[Dict[str, int]]
    labels_text: List[str]
    difficult: List[int]
    truncated: List[int]

class ObjectsInfo(NamedTuple):
    bboxes: List[Dict[str, int]]
    difficult: List[int]
    truncated: List[int]
