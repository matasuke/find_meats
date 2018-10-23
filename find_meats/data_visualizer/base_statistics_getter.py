from pathlib import Path
from typing import (
    Callable, Union, List,
    Tuple, NamedTuple, Dict, Optional
)
from tqdm import tqdm

ANNOTATION_DIR = 'Annotations'
IMAGE_DIR = 'JPEGImages'

# bounding box info.
BBOX_X_MIN = 'xmin'
BBOX_X_MAX = 'xmax'
BBOX_Y_MIN = 'ymin'
BBOX_Y_MAX = 'ymax'

# alias for typing
Coor = Optional[Dict[str, int]]
CoorList = List[Coor]
ImgShape = Optional[Tuple[int, int, int]]

# replacement of undefined name
UNDEFINED_NAME = 'UNDEFINED'

class BaseStatisticsGetter:
    __slots__ = [
        '_labels_text',
        '_filenames',
        '_label2filenames',
        '_label2objects_num',
        '_filename2objects',
        '_filename2shape',
        '_filename2annot_path',
        '_filename2img_path',
    ]

    def __init__(
            self,
            labels_text: List[str],
            filenames: List[str],
            label2filenames: Dict[str, List[str]],
            label2objects_num: Dict[str, int],
            filename2objects: Dict[str, List['ObjectInfo']],
            filename2shape: Dict[str, ImgShape],
            filename2annot_path: Dict[str, str],
            filename2img_path: Dict[str, str],
    ) -> None:
        self._labels_text = labels_text
        self._filenames = filenames
        self._label2filenames = label2filenames
        self._label2objects_num = label2objects_num
        self._filename2objects = filename2objects
        self._filename2shape = filename2shape
        self._filename2annot_path = filename2annot_path
        self._filename2img_path = filename2img_path

    @property
    def images_num(self) -> int:
        '''
        get the total number of images.
        '''
        return len(self._filenames)

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
            total_difficult += sum(self.filename2difficult(filename))

        return total_difficult

    @property
    def truncated_num(self) -> int:
        '''
        get the total number of truncated.
        '''
        total_truncated = 0
        for filename in self._filenames:
            total_truncated += sum(self.filename2truncated(filename))

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
            img_format: str,
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
        filename2objects: Dict[str, List[ObjectInfo]] = {}
        filename2shape: Dict[str, ImgShape] = {}
        filename2annot_path: Dict[str, str] = {}
        filename2img_path: Dict[str, str] = {}

        for annot in tqdm(annot_files):
            file_name, shape, bboxes, labels_text, difficult, truncated = \
                process_annot_fn(annot)

            # add each filenames.
            filenames.append(file_name)

            # add filename2shape
            filename2shape[file_name] = shape

            # count the number of objects tagged by label.
            for label_text in labels_text:
                if label_text in label2objects_num:
                    label2objects_num[label_text] += 1
                else:
                    label2objects_num[label_text] = 1

            # count the number of images taggeed by label.
            for label_text in set(labels_text):
                if label_text in label2filenames:
                    label2filenames[label_text].append(file_name)
                else:
                    label2filenames[label_text] = [file_name]

            # label2annot_path
            filename2annot_path[file_name] = str(annot)

            # label2img_path
            filename2img_path[file_name] = cls._annot2img(annot, img_format)

            # create statistics information about bboxes.
            for label_text, bbox, diff, trunc in zip(labels_text, bboxes, difficult, truncated):
                object_info = ObjectInfo(label_text, bbox, diff, trunc)

                if file_name in filename2objects:
                    filename2objects[file_name].append(object_info)
                else:
                    filename2objects[file_name] = [object_info]

        labels_text = list(label2objects_num.keys())
        return cls(
            labels_text,
            filenames,
            label2filenames,
            label2objects_num,
            filename2objects,
            filename2shape,
            filename2annot_path,
            filename2img_path,
        )

    @classmethod
    def _annot2img(
            cls,
            annot_path: Union[Path, str],
            img_format: str,
    ) -> str:
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
        img_raw_path = str(img_path)

        return img_raw_path

    @classmethod
    def _img2annot(
            cls,
            img_path: Union[str, Path],
            annot_format: str,
    ) -> str:
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
        annot_raw_path = str(annot_path)

        return annot_raw_path

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
        return len(self._label2filenames[label])

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

    def filename2image_path(self, filename: str) -> str:
        '''
        get the image path from filename.

        :param filename: filename for getting image path.
        :return: path to image.
        '''
        return self._filename2img_path[filename]

    def filename2annotation_path(self, filename: str) -> str:
        '''
        get the image path from filename.

        :param filename: filename for getting annotation path.
        :return: path to image.
        '''
        return self._filename2annot_path[filename]

    def filename2objects(self, filename: str) -> List['ObjectInfo']:
        '''
        get the labels specified in the designated annotation.

        :param filename: filename for getting object information.
        :return: list of objectinfo.
        '''
        objects_info = self._filename2objects[filename]

        return objects_info

    def filename2shape(self, filename: str) -> ImgShape:
        '''
        get the image shape from specified filename.

        :param filename: filename for getting object information.
        :return: image shape
        '''
        return self._filename2shape[filename]

    def filename2labels(self, filename: str) -> List[str]:
        '''
        get the labels specified in the designated annotation.

        :param filename: filename for getting object information.
        :return: list of labels.
        '''
        labels_text_list = []
        objects_info = self._filename2objects[filename]
        for obj_info in objects_info:
            labels_text_list.append(obj_info.label_text)

        return labels_text_list

    def filename2bboxes(self, filename: str) -> CoorList:
        '''
        get bboxes specified in the designamted annotation.

        :param filename: filename for getting bboxes.
        :return: dict of label_text and bbox.
        '''

        bboxes = []
        objects_info = self._filename2objects[filename]
        for obj_info in objects_info:
            bboxes.append(obj_info.bbox)

        return bboxes

    def filename2difficult(self, filename: str) -> List[bool]:
        '''
        get difficult specified in the designated annotation.

        :param filename: filename for getting difficult.
        :return: list of difficult in the image.
        '''
        objects_info: List[ObjectInfo] = self._filename2objects[filename]

        difficult_list = []
        for obj_info in objects_info:
            difficult_list.append(obj_info.difficult)

        return difficult_list

    def filename2truncated(self, filename: str) -> List[bool]:
        '''
        get truncated specified in the designated annotation.

        :param filename: filename for getting truncated.
        :return: list of truncated in the image.
        '''
        objects_info: List[ObjectInfo] = self._filename2objects[filename]

        truncated_list = []
        for obj_info in objects_info:
            truncated_list.append(obj_info.truncated)

        return truncated_list

    def absolute2relative_bbox_position(
            self,
            bboxes: List[Coor],
            img_shape: List[ImgShape],
            precision: int=3,
    ) -> List[Dict[str, float]]:
        '''
        convert absolute position of bbox into relative position.

        :param bboxes: list of bounding boxes.
        :param img_shape: image shape of target image.
        :param precision: the nuber of decimal.
        :return: converted bounding boxes, which are relative position in an image.
        '''
        img_width, img_height, img_channel = img_shape

        relative_bboxes = []
        for bbox in bboxes:
            relative_bboxes.append(
                {
                    BBOX_X_MIN: round(bbox[BBOX_X_MIN] / img_width, precision),  # type: ignore
                    BBOX_X_MAX: round(bbox[BBOX_X_MAX] / img_width, precision),  # type: ignore
                    BBOX_Y_MIN: round(bbox[BBOX_Y_MIN] / img_height, precision),  # type:  ignore
                    BBOX_Y_MAX: round(bbox[BBOX_Y_MAX] / img_height, precision),  # type:  ignore
                }
            )

        return relative_bboxes

class AnnotationInfo(NamedTuple):
    filename: str
    shape: ImgShape
    bboxes: CoorList
    labels_text: List[str]
    difficult: List[int]
    truncated: List[int]

class ObjectInfo(NamedTuple):
    label_text: str
    bbox: Coor
    difficult: bool
    truncated: bool
