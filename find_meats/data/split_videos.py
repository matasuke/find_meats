from argparse import ArgumentParser
import imageio
import math
from pathlib import Path
import skimage.transform
from typing import List, Tuple, Union
from tqdm import tqdm


DEFAULT_FRAME_SIZE = (224, 224)  # width, height
DEFAULT_FRAME_SHAPE = (
    DEFAULT_FRAME_SIZE[0],
    DEFAULT_FRAME_SIZE[1],
)  # height, width, channl

DEFAULT_SUFFIX = ['.mp4', '.MOV', '.avi', '.wmv']

def split_video(
        source_path: Union[str, Path],
        target_dir_path: Union[str, Path],
        num_fps: float=1,
        resize: Union[Tuple, None]=DEFAULT_FRAME_SHAPE,
) -> None:
    '''
    split a video into frames designated fps each.

    :param source_path: source video path
    :param target_dir_path: directory path to output each frames.
    :param base directory to be removed from target directory.
    :param num_fps: the number of frames to be fetched frame per seconds.
    :param resize: target frame size, if None, it doesn't change size of it.
    '''
    source_path = Path(source_path)
    target_dir_path = Path(target_dir_path) / source_path.stem

    assert source_path.exists()
    assert resize is None or isinstance(resize, tuple)
    if isinstance(resize, tuple):
        assert len(resize) == 3
    if not target_dir_path.exists():
        target_dir_path.mkdir(parents=True)

    video_frames = imageio.get_reader(source_path.as_posix(), 'ffmpeg')
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

def get_all_files(
        base_dir: Union[str, Path],
        allowed_suffix: List[str]=DEFAULT_SUFFIX,
) -> List[Path]:
    '''
    get all directories recursively.

    :param base_dir: base directory to search sub directories.
    :param allowed_suffix: suffix to be allowd as a video file.
    '''
    base_dir = Path(base_dir)
    video_path_list: List[Path] = []
    for p in base_dir.glob('*'):
        if p.is_dir():
            subvideo_path_list = get_all_files(p)
            for sp in subvideo_path_list:
                video_path_list.append(sp)
        else:
            if p.suffix in allowed_suffix:
                video_path_list.append(p)
    return video_path_list


if __name__ == '__main__':
    parser = ArgumentParser(description='split video into frames')
    parser.add_argument('--source_path', metavar='str', type=str,
                        default='./data/movies/')
    parser.add_argument('--target_path', metavar='str', type=str,
                        default='./data/processed/')
    parser.add_argument('--label_dir', metavar='str', type=str,
                        default='./data/labels/')
    parser.add_argument('--width', metavar='INT', type=int, default=None)
    parser.add_argument('--height', metavar='INT', type=int, default=None)
    parser.add_argument('--num_fps', metavar='FLOAT', type=float, default=0.5)
    args = parser.parse_args()

    if args.width and args.height:
        resize: Union[Tuple, None] = (args.width, args.height)
    else:
        resize = None

    # place all processed files into appropriated sub directories in target_path.
    source_path = Path(args.source_path)
    all_video_files = get_all_files(source_path) if source_path.is_dir() else [source_path]
    if source_path.is_dir():
        all_target_dirs = [
            args.target_path / video_files.relative_to(source_path).parent for video_files in all_video_files
        ]
    else:
        all_target_dirs = [args.target_path]

    # create label directories.
    if args.label_dir is not None:
        all_label_dirs = [
            Path(args.label_dir) / video_files.relative_to(source_path).with_suffix("")
            for video_files in all_video_files
        ]
        for label_dir in all_label_dirs:
            if not label_dir.exists():
                label_dir.mkdir(parents=True)

    for video_file, target_dir in zip(all_video_files, all_target_dirs):
        split_video(video_file, target_dir, args.num_fps, resize)
