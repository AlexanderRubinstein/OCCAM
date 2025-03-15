import json
import torch
# import h5py
import os
import sys
# import torch
# import numpy as np
# import random
# import torchvision
from datasets import load_dataset
import PIL
from stuned.utility.utils import (
    # show_images,
    # load_from_pickle,
    # append_dict,
    # get_project_root_path,
    get_with_assert,
    read_json
)


IMAGENET_9_SUBSETS = ["mixed_rand"]


# IMAGENET_D_SUBSETS = ['background', 'texture', 'material']
# JSON_PATH = os.path.join(
#     get_project_root_path(),
#     "json"
# )
# IMAGENET_D_ID_MAP_JSON = os.path.join(JSON_PATH, "imgnet_d2imgnet_id.json")


sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "src"
    )
)
from densifier.datasets.utils import (
    JSON_PATH,
    make_to_classes_mapping,
    make_model_classes_wrapper,
    get_collate_fn_in_d,
    torch_max_func
)
sys.path.pop(0)


IMAGENET_9_MAP_JSON = os.path.join(JSON_PATH, "in2in9.json")
IN9_CATEGORIES = [
    "dog",
    "bird",
    "wheeled vehicle",
    "reptile",
    "carhivore",
    "insect",
    "musical instrument",
    "primate",
    "fish"
]


class ImageNet9Dataset(torch.utils.data.Dataset):

    def __init__ (self,
        test_base_dir,
        few_test=None,
        transform=None,
        # center_crop=False,
        # to_map_labels=True
    ):
        super().__init__()

        self.test_path = test_base_dir

        self.few_test = few_test

        self.transforms=transform

        # with open(IMAGENET_9_MAP_JSON) as f:
        #     dict_in2in9 = json.load(f)

        # _, _, category2id = get_in_d_category_list()

        # category2id_patched = {
        #     k.replace(' ', '_').replace('(', '').replace(')', ''): v
        #         for k, v
        #             in category2id.items()
        # }

        # categories_list = os.listdir(self.test_path)
        # int_to_category = {
        #     int(category.split('_')): category
        #         for category
        #             in (categories_list)
        # }

        # ?? map numbers to category names
# ?? many to 1 relationship
#         dict_in9_to_in = {
#             int_to_category[v]: k
#                 for k, v
#                     in dict_in2in9.items()
#                         if v != -1
#         }

        categories_list = os.listdir(self.test_path)

        category_to_label = {
            category: int(category.split('_')[0])
                for category
                    in (categories_list)
        }

        # categories_list = os.listdir(dataset_path)
        # categories_list.sort()

        # self.categories_list = os.listdir(self.test_path)
        # self.categories_list.sort()

        # self.file_lists = []
        # self.label_lists = []

        # for each in self.categories_list:
        #     folder_path = os.path.join(self.test_path, each)

        #     files_names = os.listdir(folder_path)

        #     for eachfile in files_names:
        #         image_path = os.path.join(folder_path, eachfile)
        #         self.file_lists.append(image_path)
        #         if to_map_labels:
        #             self.label_lists.append(self.dict_imgnet_d2imagenet_id[each]+[-1]*(10-len(self.dict_imgnet_d2imagenet_id[each])))
        #         else:

        #             self.label_lists.append([int(category2id_patched[each])])
        self.file_lists, self.label_lists = make_file_and_label_lists(
            self.test_path,
            category_to_label
        )

    def __len__(self):
        if self.few_test is not None:
            return self.few_test
        else:
            return len(self.label_lists)

    def _transform(self, sample):
        return self.transforms(sample)

    def __getitem__(self, item):
        path_list=self.file_lists[item]
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
        IN9Categories(),
        aggregation_function=torch_max_func
    )
    # mapper.categories = list(range(len(IN9_CATEGOREIS)))
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


# def get_collate_fn_in_d(drop_paths):
#     def collate_fn_in_d(examples):
#         images = []
#         labels = []
#         paths = []
#         for example in examples:
#             images.append(example["images"])
#             # we take only first label as we are not interested in top-5 accuracy
#             labels.append(torch.tensor(example["labels"][0], dtype=torch.long))
#             paths.append(example["path"])
#         if drop_paths:
#             return torch.stack(images), torch.stack(labels)
#         else:
#             return torch.stack(images), torch.stack(labels), paths

#     return collate_fn_in_d


# def get_in_d_category_list():
#     with open(os.path.join(get_project_root_path(), 'json', 'imgnet_d_dir2imgnet_d_id.json')) as f:
#         category_mapping = json.load(f)
#         sorted_categories = sorted(category_mapping.values(), key=lambda value: value[0])
#         category_list = [convert_folder_name_to_category_name(value[1]) for value in sorted_categories]
#     id2category = {key: value for key, value in enumerate(category_list)}
#     category2id = {value: key for key, value in id2category.items()}
#     return category_list, id2category, category2id

def get_in_9_category_list():
    return IN9_CATEGORIES


def make_in9_wrapper(model):
    return make_model_classes_wrapper(model, make_in9_mapper)


# def convert_folder_name_to_category_name(folder_name):
#     return folder_name.replace('_', ' ').replace('-', ' ').replace('/', ' or ').lower()


def make_path2label_in_9(dataset_path, to_map_labels=True): # for counter animal
    # dataset_path = "/home/oh/arubinstein17/github/densification/data/CounterAnimal/symlinked/counter"
    transform = None
    # return_path = True
    # masks = None
    # mask_transform = None

    # dataset = CustomImageFolder(
    #     dataset_path,
    #     transform=transform,
    #     return_path=return_path,
    #     masks=masks,
    #     mask_transform=mask_transform
    # )
    dataset = ImageNet9Dataset(test_base_dir=dataset_path, transform=transform)

    # res = []
    # for item in tqdm(dataset):
    #     res.append([item[2], item[1]])
    # return res
    # return dataset.samples

    # label_lists is of form: [909, -1, -1, -1, -1, -1, -1, -1, -1, -1], so we take only first element
    return list(zip(dataset.file_lists, [el[0] for el in dataset.label_lists]))


def get_imagenet_9_dataloaders(
    # train_batch_size,
    eval_batch_size,
    dataset_config,
    num_workers,
    eval_transform,
    shuffle=False
    # to_map_labels=True,
    # logger
):

    data_dir = get_with_assert(dataset_config, "data_dir")
    # dl_types = dataset_config.get("dl_types")
    # if dl_types is not None:
    #     dl_types = set(dl_types)
    # dataloaders = {}
    # dataloader_name
    dataloader = None
    dataloader_name = None
    for dataloader_name in IMAGENET_9_SUBSETS:
        if dataloader_name == os.path.basename(os.path.dirname(data_dir)):
        # if dl_types is not None and dataloader_name not in ind_types:
        #     continue
            dataloader = torch.utils.data.DataLoader(
                # ImageNetDLoader(
                #     os.path.join(data_dir, dataloader_name),
                #     transform=eval_transform,
                #     to_map_labels=to_map_labels
                # ),
                ImageNet9Dataset(
                    test_base_dir=data_dir,
                    transform=eval_transform,
                ),
                batch_size=eval_batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=get_collate_fn_in_d(drop_paths=True)
            )
            break
    assert dataloader is not None
    assert dataloader_name is not None

    return dataloader, dataloader_name
