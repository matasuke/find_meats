pipenv run python ./find_meats/data/create_meat_master_tf_record.py \
        --dataset_dir data/MEAT_MASTER_2018/train \
        --output_dir ./data/object_detection_api/data/ \
        --output_name meat_master_2018 \
        --label_map_path data/detector/data/meat_master_label_map.pbtxt \
        --examples_path data/MEAT_MASTER_2018/train/meat_master_2018_train.txt \
        --val_ratio 0.05
