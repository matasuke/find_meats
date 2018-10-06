from argparse import ArgumentParser
import imageio
import math
import numpy as np
from pathlib import Path
import skimage.transform
from typing import List, Sequence, Mapping, Optional, Dict, Tuple
from tqdm import tqdm
DEFAULT_FRAME_SIZE = (224, 224)  # width, height
DEFAULT_FRAME_SHAPE = (
    DEFAULT_FRAME_SIZE[0],
    DEFAULT_FRAME_SIZE[1],
)  # height, width, channl


def split_movie(
        source_path: str,
        target_dir_path: str,
        resize: Tuple=None,
        num_fps: int=1,
) -> None:
    '''
    split a video into frames designated fps each.

    :param source_path: source video path
    :param target_dir_path: directory path to output each frames.
    :param resize: target frame size, if None, it doesn't change size of it.
    :param num_fps: the number of frames to be fetched frame per seconds.
    '''
    assert Path(source_path).exists()
    assert resize is None or isinstance(resize, tuple)
    if isinstance(resize, tuple):
        assert len(resize) == 3

    target_dir_path = Path(target_dir_path)
    if not target_dir_path.exists():
        target_dir_path.mkdir(parents=True)

    video_frames = imageio.get_reader(source_path, 'ffmpeg')
    total_frames = int(video_frames.get_meta_data()['nframes'])
    frame_size = resize if resize is not None else video_frames.get_meta_data()['source_size']
    frame_shape = (frame_size[0], frame_size[1], 3)
    video_fps = video_frames.get_meta_data()['fps']
    num_skip_frames = math.floor(video_fps / num_fps)

    # fetched frames
    frame_indice = [frame_index for frame_index in range(0, total_frames, num_skip_frames)]

    # error collection to get non-existance frame.
    if frame_indice[-1] > total_frames:
        frame_indice[-1] = total_frames - 1

    for frame_index in tqdm(frame_indice):
        frame = video_frames.get_data(frame_index)
        if resize is not None:
            frame = skimage.transform.resize(
                image=frame,
                output_shape=frame_shape,
                mode='reflect',
                anti_aliasing=True
            ).astype('float32')

        save_path = target_dir_path / '{0}_{1}.jpg'.format(target_dir_path.name, frame_index)
        imageio.imwrite(save_path, frame)


if __name__ == '__main__':
    parser = ArgumentParser(description='split video into frames')
    parser.add_argument('--source_path', metavar='str', type=str, required=True)
    parser.add_argument('--target_dir', metavar='str', type=str, required=True)
    parser.add_argument('--width', metavar='INT', type=int, default=None)
    parser.add_argument('--height', metavar='INT', type=int, default=None)
    parser.add_argument('--num_fps', metavar='INT', type=int, default=1)
    args = parser.parse_args()

    if args.width and args.height:
        resize = (args.width, args.height)
    else:
        resize = None

    split_movie(args.source_path, args.target_dir, resize, args.num_fps)
