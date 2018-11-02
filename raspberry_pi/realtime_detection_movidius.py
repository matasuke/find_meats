from argparse import ArgumentParser
from typing import Union
from pathlib import Path
from mvnc import mvncapi as mvnc
import sys
import numpy as np
import cv2

from find_meats.util import config_loader
from find_meats.util.preprocessor import preprocess_img

WIDTH = 300
HEIGHT = 300
MIN_SCORE_PERCENT = 60


def overlay_on_image(display_image, labels, object_info):
    source_image_width = display_image.shape[1]
    source_image_height = display_image.shape[0]

    base_index = 0
    class_id = object_info[base_index + 1]
    percentage = int(object_info[base_index + 2] * 100)
    if (percentage <= MIN_SCORE_PERCENT):
        return

    label_text = labels[int(class_id)] + " (" + str(percentage) + "%)"
    box_left = int(object_info[base_index + 3] * source_image_width)
    box_top = int(object_info[base_index + 4] * source_image_height)
    box_right = int(object_info[base_index + 5] * source_image_width)
    box_bottom = int(object_info[base_index + 6] * source_image_height)

    box_color = (255, 128, 0)  # box color
    box_thickness = 2
    cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

    scale_max = (100.0 - MIN_SCORE_PERCENT)
    scaled_prob = (percentage - MIN_SCORE_PERCENT)
    scale = scaled_prob / scale_max

    # draw the classification label string just above and to the left of the rectangle
    # label_background_color = (70, 120, 70)  # greyish green background for text
    label_background_color = (0, int(scale * 175), 75)
    label_text_color = (255, 255, 255)  # white text

    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    label_left = box_left
    label_top = box_top - label_size[1]
    if (label_top < 1):
        label_top = 1
    label_right = label_left + label_size[0]
    label_bottom = label_top + label_size[1]
    cv2.rectangle(display_image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                  label_background_color, -1)

    # label text above the box
    cv2.putText(
        display_image,
        label_text,
        (label_left, label_bottom),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        label_text_color,
        1
    )

    # display text to let user know how to quit
    cv2.rectangle(display_image, (0, 0), (100, 15), (128, 128, 128), -1)
    cv2.putText(display_image, "Q to Quit", (10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

def run_inference(np_img, labels, graph):
    assert isinstance(np_img, np.ndarray)

    resized_img = preprocess_img(np_img, (WIDTH, HEIGHT))
    graph.LoadTensor(resized_img.astype(np.float16), None)

    output, userobj = graph.GetResult()

    num_valid_boxes = int(output[0])
    for box_index in range(num_valid_boxes):
        base_index = 7 + box_index * 7
        if (not np.isfinite(output[base_index]) or
                not np.isfinite(output[base_index + 1]) or
                not np.isfinite(output[base_index + 2]) or
                not np.isfinite(output[base_index + 3]) or
                not np.isfinite(output[base_index + 4]) or
                not np.isfinite(output[base_index + 5]) or
                not np.isfinite(output[base_index + 6])):
            # boxes with non finite (inf, nan, etc) numbers must be ignored
            continue

        # x1 = max(int(output[base_index + 3] * np_img.shape[0]), 0)
        # y1 = max(int(output[base_index + 4] * np_img.shape[1]), 0)
        # x2 = min(int(output[base_index + 5] * np_img.shape[0]), np_img.shape[0]-1)
        # y2 = min((output[base_index + 6] * np_img.shape[1]), np_img.shape[1]-1)

        # overlay boxes and labels on to the image
        overlay_on_image(labels, np_img, output[base_index:base_index + 7])

def main(
        model_path: Union[str, Path],
        label_path: Union[str, Path],
        mvnc_device: int=0,
        video_device: int=0,
        window_name: str='test_detection',
):
    '''
    real time detection using movidius neural computing stick.

    :param model_path: mvnc format model path which is converted from tensorflow format.
    :param label_path: path to label name.
    '''
    if isinstance(model_path, str):
        model_path = Path(model_path)
    if isinstance(label_path, str):
        label_path = Path(label_path)
    assert model_path.exists()
    assert label_path.exists()

    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print('No devices found')
        sys.exit(1)

    device = mvnc.Device(devices[mvnc_device])
    device.OpenDevice()

    with model_path.open('rb') as f:
        graphfile = f.read()

    graph = device.AllocateGraph(graphfile)

    labels = config_loader.load(label_path)

    cap = cv2.VideoCapture(video_device)
    while True:
        ret, frame = cap.read()
        run_inference(frame, labels, graph)
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) == 27:
            break

    graph.DeallocateGraph()
    device.CloseDevice()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', metavar='str', type=str, required=True)
    parser.add_argument('--label_path', metavar='str', type=str, required=True)
    parser.add_argument('--mvnc_device', metavar='INT', type=int, default=0)
    parser.add_arguemnt('--video_device', metavar='INT', type=int, default=0)
    args = parser.parse_args()

    main(args.model_path, args.label_path, args.mvnc_path, args.video_device)
