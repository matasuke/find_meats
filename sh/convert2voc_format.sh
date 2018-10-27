# PATH_TO_MEAT_MASTER_2018_ORIGINAL_PATH has to be set as path to meat master 2018 original dataset.
pipenv run python ./find_meats/data/convert2voc_format.py \
        --dataset_name MEAT_MASTER_2018 \
        --output_dir ./data \
        --dataset_dir "PATH_TO_MEAT_MASTER_2018_ORIGINAL_PATH" \
        --name_indexing
