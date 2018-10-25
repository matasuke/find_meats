import hashlib
from pathlib import Path
import logging
from lxml import etree
from tqdm import tqdm
from typing import List, Union, Dict
import tensorflow as tf
from object_detection.utils import dataset_util, label_map_util

from find_meats.util.preprocessor import train_test_split

ANNOTATION_DIR = 'Annotations'
IMAGE_DIR = 'JPEGImages'
ANNOTATION_FORMAT = '.xml'
IMAGE_FORMAT = b'jpg'

flags = tf.app.flags
flags.DEFINE_string('dataset_dir', 'data/MEAT_MASTER_2018', 'Path to the dataset directory')
flags.DEFINE_string('output_dir', 'data/', 'Path to output directory')
flags.DEFINE_string('output_name', 'meat_master', 'Path to output directory')
flags.DEFINE_string('label_map_path', '', 'Path to label map proto')
flags.DEFINE_string('examples_path', '', 'Path to examples')
flags.DEFINE_float('val_ratio', '0.05', 'dataset ratio to be used for validation')
FLAGS = flags.FLAGS

def dict_to_tf_example(
        example_dict: Dict,
        label_map_dict: Dict[str, int],
        images_dir: Union[str, Path],
):
    if isinstance(images_dir, str):
        images_dir = Path(images_dir)
    images_dir.exists()

    width = int(example_dict['size']['width'])
    height = int(example_dict['size']['height'])

    filename = example_dict['filename']
    image_path = images_dir / filename
    raw_image_path = str(image_path)

    with tf.gfile.GFile(raw_image_path, 'rb') as fid:
        encoded_image_data = fid.read()
    key = hashlib.sha256(encoded_image_data).hexdigest()

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    for obj in example_dict['object']:
        xmins.append(float(obj['bndbox']['xmin']) / width)
        ymins.append(float(obj['bndbox']['ymin']) / height)
        xmaxs.append(float(obj['bndbox']['xmax']) / width)
        ymaxs.append(float(obj['bndbox']['ymax']) / height)
        class_name = obj['name']
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[class_name])

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
                'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
                'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
                'image/encoded': dataset_util.bytes_feature(encoded_image_data),
                'image/format': dataset_util.bytes_feature(IMAGE_FORMAT),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
            }
        )
    )

    return tf_example

def create_tf_record(
        output_path: Union[str, Path],
        annotation_dir: Union[str, Path],
        images_dir: Union[str, Path],
        label_map_dict: Dict[str, int],
        examples: List[str],
) -> None:
    if isinstance(output_path, str):
        output_path = Path(output_path)
    if isinstance(annotation_dir, str):
        annotation_dir = Path(annotation_dir)
    if isinstance(images_dir, str):
        images_dir = Path(images_dir)
    assert annotation_dir.exists()
    assert images_dir.exists()

    raw_output_path = str(output_path)

    writer = tf.python_io.TFRecordWriter(raw_output_path)

    for idx, example in tqdm(enumerate(examples)):
        annotation_filename = Path(example).with_suffix(ANNOTATION_FORMAT)
        xml_path = annotation_dir / annotation_filename
        raw_xml_path = str(xml_path)

        if not xml_path.exists():
            logging.warning('%s is not found. ignoring example.', raw_xml_path)
            continue

        with tf.gfile.GFile(raw_xml_path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        example_dict = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        tf_example = dict_to_tf_example(example_dict, label_map_dict, images_dir)
        writer.write(tf_example.SerializeToString())

    writer.close()

def main(_):
    dataset_dir = FLAGS.dataset_dir
    annotations_dir = Path(dataset_dir, ANNOTATION_DIR)
    images_dir = Path(dataset_dir, IMAGE_DIR)
    output_dir = Path(FLAGS.output_dir)
    train_output_path = Path(output_dir, f'{FLAGS.output_name}_train.record')
    val_output_path = Path(output_dir, f'{FLAGS.output_name}_val.record')
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    examples_path = Path(FLAGS.examples_path)

    examples = dataset_util.read_examples_list(str(examples_path))
    train_examples, val_examples = train_test_split(examples, test_ratio=FLAGS.valid_ratio)

    create_tf_record(train_output_path, annotations_dir, images_dir, label_map_dict, train_examples)
    create_tf_record(val_output_path, annotations_dir, images_dir, label_map_dict, val_examples)


if __name__ == '__main__':
    tf.app.run()
