python3 convert_weight.py -cf ./checkpoint/yolov3.ckpt-2500 -nc 7 -ap ./constructionsite_dataset/constructionsite_anchors.txt --freeze
python3 quick_test.py
CUDA_VISIBLE_DEVICES=3 python3 evaluate.py

