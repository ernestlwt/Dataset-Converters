import fiftyone as fo

name = "ade20k_coco_fiftyone"

dataset_dir = "/home/ernestlwt/data/ade20k_coco_fiftyone"

dataset_type = fo.types.COCODetectionDataset

dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=dataset_type,
    name=name
)

# dataset = fo.load_dataset(name)

session = fo.launch_app(dataset)
session.wait()