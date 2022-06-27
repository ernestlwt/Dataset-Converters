import json
import os

import cv2

from dataset_converters.ConverterBase import ConverterBase


class YOLO2COCOConverter(ConverterBase):

    formats = ['YOLO20212COCO']

    def __init__(self, copy_fn):
        ConverterBase.__init__(self, copy_fn)

    def _create_labels(self, required_labels):
        labels = []
        for i, line in enumerate(required_labels):
            labels.append({'supercategory': 'none', 'id': i+1, 'name': line})

        return labels

    def _read_annotations(self, input_folder, img_list): 
        instances = {
            "root": os.path.dirname(img_list[0]),
            "imgs": {}
        }

        for line in img_list:
            filename = os.path.basename(line)
            label_file = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(input_folder, instances["root"], label_file) 
            with open(label_path, 'r') as f:
                anns = [l.strip().split() for l in f.readlines()]

            instances["imgs"][filename] = []
            for ann in anns:
                obj_class = int(ann[0])
                if obj_class == 0: 
                    bbox = [float(s) for s in ann[1:]]  # bbox in yolo format: <x_center> <y_center> <width> <height> in range 0..1 
                    instances["imgs"][filename].append({'class': obj_class + 1, 'bbox': bbox})
                elif obj_class == 2:
                    bbox = [float(s) for s in ann[1:]] 
                    instances["imgs"][filename].append({'class': obj_class, 'bbox': bbox})  
                else:
                    continue # filter out stairs detections


            if not instances["imgs"][filename]: # filter out images with only stairs detections
                os.remove(label_path) # remove txt file
                os.remove(os.path.join(input_folder, instances["root"], filename)) # remove jpg file
                del instances["imgs"][filename]

        return instances

    def _yolo_bbox_to_coco(self, yolo_bbox, img_w, img_h):
        x_center, y_center, w, h = yolo_bbox
        x_center *= img_w
        y_center *= img_h
        w *= img_w
        h *= img_h

        x = max(x_center - w / 2, 0.0)
        y = max(y_center - h / 2, 0.0)

        return list(map(int, [x, y, w, h]))


    def _process_folder(self, input_folder, img_list):
        to_dump = {'images': [], 'type': 'instances', 'annotations': [], 'categories': self.labels}
        
        image_counter = 1
        instance_counter = 1
        instances = self._read_annotations(input_folder, img_list)
        img_root = instances["root"]

        folder = os.path.dirname(img_list[0])
        image_folder = os.path.join(self.output_folder, folder)
        self._ensure_folder_exists_and_is_clear(image_folder)

        for filename, anns in instances["imgs"].items():
            full_image_path = os.path.join(input_folder, img_root, filename)
            image = cv2.imread(full_image_path)
            to_dump['images'].append(
                {
                    'file_name': filename,
                    'height': image.shape[0],
                    'width': image.shape[1],
                    'id': image_counter
                }
            )
            for ann in anns:
                bbox = self._yolo_bbox_to_coco(ann["bbox"], image.shape[1], image.shape[0])
                x, y, w, h = bbox

                if any([b < 0 for b in bbox]):
                    print("Point 2", bbox, ann["bbox"])

                xmin = x
                xmax = x + w
                ymin = y
                ymax = y + h
                to_dump['annotations'].append(
                    {
                        'segmentation': list(map(int,[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])),
                        'area': w * h,
                        'iscrowd': 0,
                        'image_id': image_counter,
                        'bbox': bbox,
                        'category_id': ann["class"],
                        'id': instance_counter,
                        'ignore': 0
                    }
                )
                instance_counter += 1
            self.copy(full_image_path, image_folder)
            image_counter += 1

        with open(os.path.join(self.annotations_folder, '{0}.json'.format(folder)), 'w') as f:
            json.dump(to_dump, f, indent=4)

    def _run(self, input_folder, output_folder, FORMAT):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.annotations_folder = os.path.join(output_folder, 'annotations')

        self._ensure_folder_exists_and_is_clear(output_folder)
        self._ensure_folder_exists_and_is_clear(self.annotations_folder)

        CLASSES = [
            "door",
            "window"
        ]

        self.labels = self._create_labels(CLASSES)

        # creates list of relative paths to each image in the dataset
        img_folder = os.path.join(input_folder, 'images')
        list_of_imgs = []
        for img_file in os.listdir(img_folder):
            if img_file.endswith(".jpg"):
                list_of_imgs.append(os.path.join('images', img_file))
        
        self._process_folder(input_folder, list_of_imgs)