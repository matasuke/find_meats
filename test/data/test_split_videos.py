import pytest
from meats_detector.data.split_videos import split_video, get_all_files

BASE_DIR = './tmp/datasource/movies/'
TEST_MOVIE_PATH = './tmp/datasource/movies/test_video_dir1/VID_20180619_212451.mp4'
TARGET_DIR = './tmp/processed/'
NUM_FPS = 1

def test_split_movie():
    EXPECTED = []
    split_video(BASE_DIR, TARGET_DIR, NUM_FPS, None)

def test_get_all_files():
    EXPECTED = []
    video_path_list = get_all_files(TEST_MOVIE_PATH)
    assert video_path_list  == EXPECTED
