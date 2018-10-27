curl -O http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz -o data/object_detection_api/data/pretrained/model.tar.gz
tar -zxvf data/object_detection_api/data/pretrained/model.tar.gz -C data/object_detection_api/data/pretrained
rm -rf data/object_detection_api/data/pretrained/model.tar.gz
