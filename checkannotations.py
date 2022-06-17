from pycocotools.coco import COCO
import numpy as np

training_annotation = COCO("./data/ade20k_coco/annotations/train.json")

cats = training_annotation.loadCats(training_annotation.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))