import os
from sklearn.model_selection import train_test_split
import shutil
import sys
from pathlib import Path
import json
home = os.path.expanduser("~")

CLASS_NAME_LABEL_MAP = {"apple": 0} # change the label and class name according to your labeling

def partition_dataset(image_dir, label_dir, main_dir):
    images = []
    # for image_file in sorted(Path(image_dir).resolve().glob("*.png")):
    #     images.append(str(image_file))
    annotations = []
    for annotation_file in sorted(Path(label_dir).resolve().glob("*.txt")):
        annotations.append(str(annotation_file))
        image_file_name = Path(annotation_file).stem + ".png"
        # print("image file name: ", image_file_name)
        image_file_path = os.path.join(image_dir, image_file_name)
        # print("image file path: ", image_file_path)
        images.append(image_file_path)
    
    # Split the dataset into train-valid-test splits 
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
    print("Number of training images: ", len(train_images))
    print("Number of validation images: ", len(val_images))
    # Move the splits into their folders
    move_files_to_folder(train_images, image_dir+'/train')
    move_files_to_folder(val_images, image_dir+'/val')
    move_files_to_folder(train_annotations, label_dir+'/train')
    move_files_to_folder(val_annotations, label_dir+'/val')

#Utility function to copy images 
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            # if already in the destination folder, skip
            if os.path.exists(os.path.join(destination_folder, Path(f).name)):
                continue
            shutil.move(f, destination_folder)
            # shutil.copyfile(f, destination_folder)
        except:
            print(f)
            assert False

def cp_images_accorading_label(label_dir, source_image_dir, target_image_dir):
    # for all the files in the label_dir, copy the corresponding image file in source_image_dir to the target_image_dir
    for annotation_file in sorted(Path(label_dir).resolve().glob("*.txt")):
        image_file_name = Path(annotation_file).stem + ".png"
        image_file_path = os.path.join(source_image_dir, image_file_name)
        shutil.copyfile(image_file_path, os.path.join(target_image_dir, image_file_name))



def json2yolo(json_file, save_dir):
    with open(json_file) as f:
        data = json.load(f)
        image_path = data["imagePath"]
        imageHeight = data["imageHeight"]
        imageWidth = data["imageWidth"]
        shapes = data["shapes"]
        # create a txt file with the same name as the image file
        txt_file = save_dir + '/' +Path(image_path).stem + ".txt"
        # if the file exist already, remove its content and write new content
        with open(txt_file, "w") as txt_f:
            for shape in shapes:
                class_name = shape["label"]
                points = shape["points"]
                if class_name in CLASS_NAME_LABEL_MAP:
                    label = CLASS_NAME_LABEL_MAP[class_name]
                    # write the label to the txt file
                    txt_f.write(f"{label}")
                    for point in points:
                        x = point[0]
                        y = point[1]
                        # normalize the x and y
                        x = x / imageWidth
                        y = y / imageHeight
                        txt_f.write(f" {x} {y}")
                    txt_f.write("\n")

def main():
    # given the labelled and images and json files are in the folder any_labelling of the main folder, create a json, image and label folder
    training_root_dir = home+"/bags/fruit_counting/image_labelling_data/2024-06-12-12-08-14_apple" # replace the folder to direct to your folder with anylabeling folder
    any_labelling_dir = os.path.join(training_root_dir, "any_labeling")
    json_dir = os.path.join(training_root_dir, "json")
    image_dir = os.path.join(training_root_dir, "images")
    label_dir = os.path.join(training_root_dir, "labels")

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    # cp all the json files in any_labelling to json folder
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    for json_file in sorted(Path(any_labelling_dir).resolve().glob("*.json")):
        if not os.path.exists(os.path.join(json_dir, json_file.name)):
            shutil.copyfile(json_file, os.path.join(json_dir, json_file.name))
        # also copy the image file to the image folder
        image_file_name = Path(json_file).stem + ".png"
        image_file_path = os.path.join(any_labelling_dir, image_file_name)
        if not os.path.exists(os.path.join(image_dir, image_file_name)):
            shutil.copyfile(image_file_path, os.path.join(image_dir, image_file_name))

    # convert the json files to yolo format
    for json_file in sorted(Path(json_dir).resolve().glob("*.json")):
        json2yolo(json_file, label_dir)

    # seperate the images and labels into train and val folders
    # make a train and val folder under images and labels
    if not os.path.exists(image_dir+'/train'):
        os.makedirs(image_dir+'/train')
    if not os.path.exists(image_dir+'/val'):
        os.makedirs(image_dir+'/val')
    if not os.path.exists(label_dir+'/train'):
        os.makedirs(label_dir+'/train')
    if not os.path.exists(label_dir+'/val'):
        os.makedirs(label_dir+'/val')
    partition_dataset(image_dir, label_dir, training_root_dir)

if __name__ == '__main__':
    main()
