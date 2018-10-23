import pytest
from find_meats.data_visualizer.voc_statistics_getter import VocStatisticsGetter
from .test_configurations import SAVE_PATH


@pytest.fixture()
def loaded_voc_getter():
    voc_statistics_getter = VocStatisticsGetter.load(SAVE_PATH)
    return voc_statistics_getter
