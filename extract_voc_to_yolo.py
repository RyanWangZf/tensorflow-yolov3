# -*- coding: utf-8 -*-
"""Extract raw VOC datasets to YOLO info files.
"""
import os
import random
from glob import glob

import xml.etree.ElementTree as ET

# defined classes in this dataset
classes = ["worker","worker_no_helmet","dump_truck","excavator","loader","concrete_mixer_truck","road_roller"]

# define train test split
test_ratio = 0.1

root_path = os.getcwd()
voc_path = os.path.join(root_path,"constructionsite_dataset/VOC2007")
anno_path = os.path.join(voc_path,"train/Annotations")
jpg_path = os.path.join(voc_path,"train/JPEGImages")

dataset_info_path = os.path.join(root_path,"constructionsite_dataset")

if not os.path.exists(dataset_info_path):
    os.makedirs(dataset_info_path)

def convert_annotation(image_name,list_file):
    img_id = os.path.basename(image_name)[:-4]
    xml_path = os.path.join(anno_path,"{}.xml".format(img_id))
    in_file = open(xml_path)
    tree = ET.parse(in_file)
    root = tree.getroot()
    for obj in root.iter("object"):
        difficult = obj.find("difficult").text
        cls = obj.find("name").text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find("bndbox")
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " +  " ".join([str(a) for a in b]) + " " + str(cls_id))
    return

def main():
    image_names = glob(os.path.join(jpg_path,"*.jpg"))
    num_test = int(len(image_names)*test_ratio)

    # write train
    train_img_names = image_names[num_test:]
    trainfile_path = os.path.join(dataset_info_path,"voc_train.txt")
    train_file = open(trainfile_path,"w")
    for i,img in enumerate(train_img_names):
        train_file.write(img)
        convert_annotation(img,train_file)
        train_file.write("\n")
    train_file.close()

    # write test
    test_img_names = image_names[:num_test]
    testfile_path = os.path.join(dataset_info_path,"voc_test.txt")
    test_file = open(testfile_path,"w")
    for i,img in enumerate(test_img_names):
        test_file.write(img)
        convert_annotation(img,test_file)
        test_file.write("\n")
    test_file.close()

    # save names
    dataset_names_path = os.path.join(dataset_info_path,"constructionsite.names")
    names = open(dataset_names_path,"w")
    for name in classes:
        names.write(name + "\n")
    names.close()

    print("Done, train: {}, test:{}".format(len(image_names)-num_test,num_test))
    return


if __name__ == '__main__':
    main()



