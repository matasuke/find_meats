from find_meats.util.preprocessor import train_test_split

def test_train_test_split_with_num():
    data_list = list(range(100))
    TEST_NUM = 40
    SPLIT_NUM = len(data_list) - TEST_NUM
    EXPECTED_TRAIN = data_list[:SPLIT_NUM]
    EXPECTED_TEST = data_list[SPLIT_NUM:]

    list_train, list_test = train_test_split(data_list, test_num=TEST_NUM, shuffle=False)

    assert list_train == EXPECTED_TRAIN
    assert list_test == EXPECTED_TEST

def test_train_test_split_with_num_and_shuffle():
    data_list = list(range(100))
    TEST_NUM = 40
    SPLIT_NUM = len(data_list) - TEST_NUM
    EXPECTED_TRAIN_NUM = 60
    EXPECTED_TEST_NUM = 40

    train_list, test_list = train_test_split(data_list, test_num=TEST_NUM, shuffle=True)

    assert len(train_list) == EXPECTED_TRAIN_NUM

def test_train_test_split_with_ratio():
    data_list = list(range(100))
    SPLIT_RATIO = 0.3
    EXPECTED_TRAIN = data_list[:70]
    EXPECTED_TEST = data_list[70:]

    list_train, list_test = train_test_split(data_list, test_ratio=SPLIT_RATIO, shuffle=False)

    assert list_train == EXPECTED_TRAIN
    assert list_test == EXPECTED_TEST

def test_train_test_split_with_ratio_and_shuffle():
    data_list = list(range(100))
    SPLIT_RATIO = 0.3
    EXPECTED_TRAIN = 70
    EXPECTED_TEST = 30

    list_train, list_test = train_test_split(data_list, test_ratio=SPLIT_RATIO, shuffle=True)

    assert len(list_train) == EXPECTED_TRAIN
    assert len(list_test) == EXPECTED_TEST
