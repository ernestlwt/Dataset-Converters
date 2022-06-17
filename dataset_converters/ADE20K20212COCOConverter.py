from dataset_converters.ConverterBase import ConverterBase

import json
import os
import re
import time

import cv2
import numpy as np

background_color = 0


class ADE20K2COCOConverter(ConverterBase):

    formats = ['ADE20K20212COCO']

    def __init__(self, copy_fn):
        ConverterBase.__init__(self, copy_fn)
        self.label_names = []

    def _get_files(self, folder):
        all_files = []
        for root, dirs, files in os.walk(folder):
            all_files.extend([os.path.join(root, file) for file in files])
        return all_files

    def _get_by_pattern(self, pattern, files):
        return [file for file in files if re.match(pattern, file) is not None]

    def _get_image_filenames(self, files):
        return self._get_by_pattern('.*jpg', files)

    def _get_segmentation_filenames(self, files):
        return self._get_by_pattern('.*_seg[.]png', files)

    def _get_annotation_filenames(self, files):
        return self._get_by_pattern('.*json', files)

    def _read_class_names(self, attribute_filename):
        with open(attribute_filename, 'r') as f:
            lines = [line.split(' # ') for line in f.readlines()]
        class_names = [line[4] for line in lines if line[1] == '0']
        return class_names

    def _read_annotation(self, annotation_filename):
        with open(annotation_filename, 'r') as f:
            annotation_json = json.load(f)
        return annotation_json

    def _get_bbox(self, seg_x, seg_y):
        x_min = float("inf")
        y_min = float("inf")
        x_max = 0
        y_max = 0
        for s_x in seg_x:
            x_min = min(x_min, s_x)
            x_max = max(x_max, s_x)
        for s_y in seg_y:
            y_min = min(y_min, s_y)
            y_max = max(y_max, s_y)

        return x_min, y_min, x_max - x_min, y_max - y_min

    def _get_area(self, seg_x, seg_y, w, h):
        image = np.zeros((w, h, 3), np.int32)
        points = np.array([list(a) for a in zip(seg_x, seg_y)], np.int32)
        image = cv2.fillPoly(image, pts=[points], color=(1,0,0))
        area = int(np.sum(image))
        return area

    def _process_folder(self, input_folder, output_images_folder, output_annotations_file, target_labels):
        self._ensure_folder_exists_and_is_clear(output_images_folder)

        files = self._get_files(input_folder)
        files.sort()

        image_filenames = self._get_image_filenames(files)
        annotation_filenames = self._get_annotation_filenames(files)

        images_count = len(image_filenames)
        assert(images_count == len(annotation_filenames))
        to_dump = {'images': [], 'type': 'instances', 'annotations': [], 'categories': []}


        label_ids = [label['id'] for label in target_labels]
        label_names = [label['label'] for label in target_labels]
        
        for new_idx, label_name in enumerate(label_names, 1):
            to_dump['categories'].append({'supercategory': 'none', 'id': new_idx, 'name': label_name})


        instance_counter = 1
        iterables = zip(image_filenames, annotation_filenames)
        for i, (image, annotation_filename) in enumerate(iterables, 1):
            try:
                annotation = self._read_annotation(annotation_filename)
            except:
                continue
            print(i)
            h, w = annotation['annotation']['imsize'][:2]

            object_found = False
            for obj in annotation['annotation']['object']:
                if obj['name_ndx'] in label_ids:
                    category_id = label_ids.index(obj['name_ndx']) + 1
                    object_found = True
                else:
                    continue # skip if object class is not wanted

                seg_x = obj['polygon']['x']
                seg_y = obj['polygon']['y']

                seg = []
                for x, y in zip(seg_x, seg_y):
                    seg.append(x)
                    seg.append(y)
                
                bbox = self._get_bbox(seg_x, seg_y)
                area = self._get_area(seg_x, seg_y, w, h)

                to_dump['annotations'].append(
                    {
                        'segmentation': seg,
                        'area': area,
                        'iscrowd': 0,
                        'image_id': i,
                        'bbox': bbox,
                        'category_id': category_id,
                        'id': instance_counter,
                        'ignore': 0
                    }
                )
                instance_counter += 1


            if object_found:
                to_dump['images'].append(
                    {
                        'file_name': os.path.basename(image),
                        'height': h,
                        'width': w,
                        'id': i
                    }
                )
                self.copy(image, output_images_folder)

        with open(output_annotations_file, 'w') as f:
            json.dump(to_dump, f)

    def _run(self, input_folder, output_folder, FORMAT):
        images_folder = os.path.join(input_folder, 'images/ADE')
        annotations_folder = os.path.join(output_folder, 'annotations')

        REQUIRED_LABELS = [
                {"id":3055, "label":"window pane"}, # WINDOWPANE
                {"id":783, "label":"double door"}, # DOUBLE DOOR
                {"id":1747, "label":"glass pane"}, # GLASS PANE
                {"id":778, "label":"door frame"}, # DOOR FRAME
                {"id":2251, "label":"screen door"}, # SCREEN DOOR
                {"id":2439, "label":"sliding door"}, # SLIDING DOOR
                {"id":754, "label":"display window"}, # DISPLAY WINDOW, SHOP WINDOW
                {"id":851, "label":"elevator"}, # ELEVATOR DOOR
                {"id":1141, "label":"grille door"}, # GRILLE DOOR
                {"id":995, "label":"folding door"}, # FOLDING DOOR
                {"id":782, "label":"dormer window"}, # DORMER WINDOW
                {"id":2765, "label":"ticket window"}, # TICKET WINDOW
                {"id":852, "label":"elevator doors"}, # ELEVATOR DOORS (SIMILAR TO 851?)
                {"id":2286, "label":"security door"}, # SECURITY DOOR
                {"id":3054, "label":"window scarf"}, # WINDOW SCARF (not window but can be used to find windows?)
                {"id":2358, "label":"shower door"}, # SHOWER DOOR
                {"id":2164, "label":"rose window"}, # ROSE WINDOW
                {"id":2287, "label":"security door frame"}, # SECURITY DOOR FRAME
                {"id":3050, "label":"window"}, # WINDOW
                {"id":774, "label":"door"}, # DOOR
                {"id":2346, "label":"shop window"}, #SHOP WINDOW
                {"id":776, "label":"door frames"}, # DOOR FRAME (SIMILAR TO 778)
                {"id":3056, "label":"windows"}, # WINDOWS (SIMILAR TO 3050)
                {"id":1062, "label":"garage door"}, # GARAGE DOOR
                {"id":2103, "label":"revolving door"}, # REVOLVING DOOR
                {"id":779, "label":"doors"}, # DOORS (SIMILAR TO 774)
            ]

        self._ensure_folder_exists_and_is_clear(output_folder)
        self._ensure_folder_exists_and_is_clear(annotations_folder)

        self._process_folder(
            os.path.join(images_folder, 'training'),
            os.path.join(output_folder, 'train'),
            os.path.join(annotations_folder, 'train.json'),
            REQUIRED_LABELS
        )
        self._process_folder(
            os.path.join(images_folder, 'validation'),
            os.path.join(output_folder, 'val'),
            os.path.join(annotations_folder, 'val.json'),
            REQUIRED_LABELS
        )
