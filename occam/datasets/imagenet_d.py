import json
import torch
import os
import sys
import PIL
from stuned.utility.utils import (
    get_with_assert,
    read_json,
)


sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
)
from occam.datasets.utils import (
    JSON_PATH,
    DATASETS_PATH,
    get_collate_fn_in_d,
    make_mapping_dict_from_folder,
)

sys.path.pop(0)


IMAGENET_D_SUBSETS = ["background", "texture", "material"]
IMAGENET_D_ID_MAP_JSON = os.path.join(JSON_PATH, "imgnet_d2imgnet_id.json")
IN_D_PATH = os.path.join(DATASETS_PATH, "ImageNet-D")


# taken from: https://github.com/chenshuang-zhang/imagenet_d/blob/main/utils/data_loaders_imgnet_id.py#L6
class ImageNetDLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        test_base_dir,
        few_test=None,
        transform=None,
        center_crop=False,
        to_map_labels=True,
    ):
        super().__init__()

        self.test_path = test_base_dir
        self.categories_list = os.listdir(self.test_path)
        self.categories_list.sort()

        self.file_lists = []
        self.label_lists = []
        self.few_test = few_test

        self.transforms = transform

        with open(IMAGENET_D_ID_MAP_JSON) as f:
            self.dict_imgnet_d2imagenet_id = json.load(f)

        _, _, category2id = get_in_d_category_list()

        category2id_patched = {
            k.replace(" ", "_").replace("(", "").replace(")", ""): v
            for k, v in category2id.items()
        }

        for each in self.categories_list:
            folder_path = os.path.join(self.test_path, each)

            files_names = os.listdir(folder_path)

            for eachfile in files_names:
                image_path = os.path.join(folder_path, eachfile)
                self.file_lists.append(image_path)
                if to_map_labels:
                    self.label_lists.append(
                        self.dict_imgnet_d2imagenet_id[each]
                        + [-1]
                        * (10 - len(self.dict_imgnet_d2imagenet_id[each]))
                    )
                else:
                    self.label_lists.append([int(category2id_patched[each])])

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


def get_in_d_category_list():

    category_mapping = read_json(
        os.path.join(JSON_PATH, "imgnet_d_dir2imgnet_d_id.json")
    )
    sorted_categories = sorted(
        category_mapping.values(), key=lambda value: value[0]
    )
    category_list = [
        convert_folder_name_to_category_name(value[1])
        for value in sorted_categories
    ]
    id2category = {key: value for key, value in enumerate(category_list)}
    category2id = {value: key for key, value in id2category.items()}
    return category_list, id2category, category2id


def convert_folder_name_to_category_name(folder_name):
    return (
        folder_name.replace("_", " ")
        .replace("-", " ")
        .replace("/", " or ")
        .lower()
    )


def make_path2label_in_d(
    dataset_path, to_map_labels=True
):
    transform = None

    dataset = ImageNetDLoader(
        test_base_dir=dataset_path,
        transform=transform,
        to_map_labels=to_map_labels,
    )

    # label_lists is of form: [909, -1, -1, -1, -1, -1, -1, -1, -1, -1], so we take only first element
    return list(zip(dataset.file_lists, [el[0] for el in dataset.label_lists]))


def get_imagenet_d_dataloaders(
    eval_batch_size,
    dataset_config,
    num_workers,
    eval_transform,
    to_map_labels=True,
    shuffle=False
):
    data_dir = get_with_assert(dataset_config, "data_dir")
    ind_types = dataset_config.get("ind_types")
    if ind_types is not None:
        ind_types = set(ind_types)
    ind_dataloaders = {}
    for dataloader_name in IMAGENET_D_SUBSETS:
        if ind_types is not None and dataloader_name not in ind_types:
            continue
        ind_dataloaders[dataloader_name] = torch.utils.data.DataLoader(
            ImageNetDLoader(
                os.path.join(data_dir, dataloader_name),
                transform=eval_transform,
                to_map_labels=to_map_labels,
            ),
            batch_size=eval_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=get_collate_fn_in_d(drop_paths=True),
        )
    return ind_dataloaders


def make_mapping_dict_imagenet_d(
    images_folder, masks_path, bboxes_path, separate_masks_folder
):
    dataset_kwargs = {}
    if isinstance(images_folder, (list, tuple)):
        images_folder, dataset_kwargs = images_folder
    path2label = make_path2label_in_d(images_folder, **dataset_kwargs)

    mapping_dict = make_mapping_dict_from_folder(
        path2label=path2label,
        masks_path=masks_path,
        bboxes_path=bboxes_path,
        separate_masks_folder=separate_masks_folder,
    )
    return mapping_dict
