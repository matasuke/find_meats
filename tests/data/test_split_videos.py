from pathlib import Path
from find_meats.data.split_videos import get_all_files

BASE_DIR = './tmp/data/movies/'
TEST_MOVIE_PATH = './tmp/data/movies/test_video_dir1/VID_20180619_212451.mp4'
TARGET_DIR = './tmp/processed/'
NUM_FPS = 1

def test_split_movie():
    # EXPECTED = []
    # split_video(BASE_DIR, TARGET_DIR, NUM_FPS, None)
    pass

def test_get_all_files():
    EXPECTED = [
        Path('tmp/data/movies/test_video_dir2/VID_20180619_212451.mp4'),
        Path('tmp/data/movies/test_video_dir1/VID_20180619_213155.mp4'),
        Path('tmp/data/movies/test_video_dir1/VID_20180619_212451.mp4'),
        Path('tmp/data/movies/test_video_dir2/test_video_sub_dir1/VID_20180619_212451.mp4'),
    ]

    video_path_list = get_all_files(BASE_DIR)
    for video_path in video_path_list:
        assert video_path in EXPECTED
