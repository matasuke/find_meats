curl http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz -o data/object_detection_api/data/pretrained/model.tar.gz
tar zxvf data/object_detection_api/data/pretrained/model.tar.gz -C data/object_detection_api/data/pretrained
rm -rf data/object_detection_api/data/pretrained/model.tar.gz
