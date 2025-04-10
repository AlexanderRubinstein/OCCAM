import torch
import os
import sys
import PIL
from stuned.utility.utils import (
    get_with_assert,
    read_json,
    get_project_root_path,
)


IMAGENET_9_SUBSETS = ["mixed_rand"]


sys.path.insert(0, get_project_root_path())
from occam.datasets.utils import (
    JSON_PATH,
    DATASETS_PATH,
    make_to_classes_mapping,
    make_model_classes_wrapper,
    get_collate_fn_in_d,
    torch_max_func,
    make_mapping_dict_generic,
)

sys.path.pop(0)


IMAGENET_9_PATH = os.path.join(DATASETS_PATH, "ImageNet-9", "mixed_rand", "val")
IMAGENET_9_MAP_JSON = os.path.join(JSON_PATH, "in2in9.json")
IN9_CATEGORIES = [
    "dog",
    "bird",
    "wheeled vehicle",
    "reptile",
    "carnivore",
    "insect",
    "musical instrument",
    "primate",
    "fish",
]


class ImageNet9Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        test_base_dir,
        few_test=None,
        transform=None,
    ):
        super().__init__()

        self.test_path = test_base_dir

        self.few_test = few_test

        self.transforms = transform

        categories_list = os.listdir(self.test_path)

        category_to_label = {
            category: int(category.split("_")[0])
            for category in (categories_list)
        }

        self.file_lists, self.label_lists = make_file_and_label_lists(
            self.test_path, category_to_label
        )

    def __len__(self):
        if self.few_test is not None:
            return self.few_test
        else:
            return len(self.label_lists)

    def _transform(self, sample):
        return self.transforms(sample)

    def __getitem__(self, item):
        path_list = self.file_lists[item]
        img = PIL.Image.open(path_list).convert("RGB")

        img_tensor = self._transform(img)
        img.close()
        labels = self.label_lists[item]

        return {"images": img_tensor, "labels": labels, "path": path_list}


def invert_dict_with_repetitions(d):
    res = {}
    for k, v in d.items():
        res.setdefault(v, [])
        res[v].append(k)
    return res


class IN9Categories:
    def __init__(self):
        map_to_in9 = read_json(IMAGENET_9_MAP_JSON)
        self.in9_to_in1000 = invert_dict_with_repetitions(map_to_in9)
        self.in9_to_in1000 = {
            k: [int(vi) for vi in v] for k, v in self.in9_to_in1000.items()
        }
        self.categories = IN9_CATEGORIES
        self.cat_to_id = {cat: i for i, cat in enumerate(self.categories)}

    def __call__(self, category_name):
        category_id = self.cat_to_id[category_name]
        return self.in9_to_in1000[category_id]


def make_in9_mapper():
    mapper = make_to_classes_mapping(
        IN9Categories(), aggregation_function=torch_max_func
    )
    return mapper


def make_file_and_label_lists(dataset_path, category_name2label):
    categories_list = os.listdir(dataset_path)
    categories_list.sort()

    file_lists = []
    label_lists = []

    for category in categories_list:
        folder_path = os.path.join(dataset_path, category)

        files_names = os.listdir(folder_path)

        for eachfile in files_names:
            image_path = os.path.join(folder_path, eachfile)
            assert os.path.isfile(image_path)
            file_lists.append(image_path)

            label_lists.append([int(category_name2label[category])])

    return file_lists, label_lists


def get_in_9_category_list():
    return IN9_CATEGORIES


def make_in9_wrapper(model):
    return make_model_classes_wrapper(model, make_in9_mapper)


def make_path2label_in_9(
    dataset_path, to_map_labels=True
):
    transform = None
    dataset = ImageNet9Dataset(test_base_dir=dataset_path, transform=transform)

    # label_lists is of form: [909, -1, -1, -1, -1, -1, -1, -1, -1, -1], so we take only first element
    return list(zip(dataset.file_lists, [el[0] for el in dataset.label_lists]))


def get_imagenet_9_dataloaders(
    eval_batch_size,
    dataset_config,
    num_workers,
    eval_transform,
    shuffle=False
):
    data_dir = get_with_assert(dataset_config, "data_dir")
    dataloader = None
    dataloader_name = None
    for dataloader_name in IMAGENET_9_SUBSETS:
        if dataloader_name == os.path.basename(os.path.dirname(data_dir)):
            dataloader = torch.utils.data.DataLoader(
                ImageNet9Dataset(
                    test_base_dir=data_dir,
                    transform=eval_transform,
                ),
                batch_size=eval_batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=get_collate_fn_in_d(drop_paths=True),
            )
            break
    assert dataloader is not None
    assert dataloader_name is not None

    return dataloader, dataloader_name


def make_mapping_dict_imagenet_9(
    images_folder, masks_path, bboxes_path, separate_masks_folder
):
    return make_mapping_dict_generic(
        images_folder,
        masks_path,
        bboxes_path=bboxes_path,
        separate_masks_folder=separate_masks_folder,
        path2label_func=make_path2label_in_9,
    )
