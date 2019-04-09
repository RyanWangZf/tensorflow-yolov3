python3 extract_voc_to_yolo.py
python3 kmeans.py --dataset_txt constructionsite_dataset/voc_train.txt --anchors_txt constructionsite_dataset/constructionsite_anchors.txt
python3 ./core/convert_tfrecord.py --dataset_txt constructionsite_dataset/voc_train.txt --tfrecord_path_prefix tfrecords/voc_train
python3 ./core/convert_tfrecord.py --dataset_txt constructionsite_dataset/voc_test.txt --tfrecord_path_prefix tfrecords/voc_test