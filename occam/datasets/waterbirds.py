# import json
# import torch
# import h5py
import os
import sys

# import torch
# import numpy as np
# import random
# import torchvision
# from datasets import load_dataset
# import PIL
# from stuned.utility.utils import (
#     show_images,
#     load_from_pickle,
#     append_dict,
#     get_project_root_path,
#     get_with_assert
# )


sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
)
from occam.datasets.utils import (
    # JSON_PATH,
    # get_collate_fn_in_d,
    make_custom_folder_path2label,
    make_mapping_dict_generic,
)
from occam.datasets.utils import (
    DATASETS_PATH,
    make_to_classes_mapping,
    make_model_classes_wrapper,
    torch_max_func,
)

sys.path.pop(0)


WATERBIRDS_BASE_PATH = os.path.join(DATASETS_PATH, "Waterbirds")
WATERBIRDS_PATHS = [
    os.path.join(WATERBIRDS_BASE_PATH, "test_split", f"group_{group_id}")
    for group_id in range(4)
]
WATERBIRDS_ONLY_FG_PATHS = [
    os.path.join(
        WATERBIRDS_BASE_PATH, "FG-Only", "test_split", f"group_{group_id}"
    )
    for group_id in range(4)
]


WATER_BIRDS_TYPES = [
    "Albatross",  # Seabirds
    "Auklet",
    "Cormorant",
    "Frigatebird",
    "Fulmar",
    "Gull",
    "Jaeger",
    "Kittiwake",
    "Pelican",
    "Puffin",
    "Tern",
    "Gadwall",  # Waterfowl
    "Grebe",
    "Mallard",
    "Merganser",
    "Guillemot",
    "Pacific_Loon",
]


LAND_BIRDS_LIST = [
    "house sparrow",
    "seaside sparrow",
    "myrtle warbler",
    "bronzed cowbird",
    "american three toed woodpecker",
    "shiny cowbird",
    "gray crowned rosy finch",
    "brown thrasher",
    "chestnut sided warbler",
    "belted kingfisher",
    "winter wren",
    "yellow bellied flycatcher",
    "bank swallow",
    "green kingfisher",
    "baltimore oriole",
    "scissor tailed flycatcher",
    "green tailed towhee",
    "scott oriole",
    "whip poor will",
    "cactus wren",
    "yellow breasted chat",
    "lazuli bunting",
    "warbling vireo",
    "european goldfinch",
    "florida jay",
    "groove billed ani",
    "cape may warbler",
    "white breasted kingfisher",
    "le conte sparrow",
    "pine warbler",
    "black billed cuckoo",
    "gray catbird",
    "red eyed vireo",
    "orchard oriole",
    "chipping sparrow",
    "rose breasted grosbeak",
    "downy woodpecker",
    "golden winged warbler",
    "carolina wren",
    "fish crow",
    "mangrove cuckoo",
    "philadelphia vireo",
    "field sparrow",
    "loggerhead shrike",
    "red bellied woodpecker",
    "hooded warbler",
    "song sparrow",
    "white eyed vireo",
    "ruby throated hummingbird",
    "blue headed vireo",
    "northern flicker",
    "prairie warbler",
    "cedar waxwing",
    "canada warbler",
    "marsh wren",
    "mourning warbler",
    "house wren",
    "barn swallow",
    "mockingbird",
    "great crested flycatcher",
    "bay breasted warbler",
    "cerulean warbler",
    "blue winged warbler",
    "cliff swallow",
    "wilson warbler",
    "bewick wren",
    "louisiana waterthrush",
    "white throated sparrow",
    "american crow",
    "indigo bunting",
    "american pipit",
    "grasshopper sparrow",
    "brewer sparrow",
    "acadian flycatcher",
    "northern waterthrush",
    "harris sparrow",
    "orange crowned warbler",
    "ovenbird",
    "brown creeper",
    "swainson warbler",
    "baird sparrow",
    "clay colored sparrow",
    "nashville warbler",
    "painted bunting",
    "yellow throated vireo",
    "summer tanager",
    "vermilion flycatcher",
    "anna hummingbird",
    "clark nutcracker",
    "ringed kingfisher",
    "rock wren",
    "american goldfinch",
    "pileated woodpecker",
    "tree sparrow",
    "red headed woodpecker",
    "white necked raven",
    "purple finch",
    "green violetear",
    "nelson sharp tailed sparrow",
    "common raven",
    "fox sparrow",
    "palm warbler",
    "gray kingbird",
    "red cockaded woodpecker",
    "white crowned sparrow",
    "boat tailed grackle",
    "black capped vireo",
    "sage thrasher",
    "great grey shrike",
    "evening grosbeak",
    "yellow billed cuckoo",
    "prothonotary warbler",
    "blue grosbeak",
    "black throated blue warbler",
    "scarlet tanager",
    "bohemian waxwing",
    "american redstart",
    "tree swallow",
    "dark eyed junco",
    "pine grosbeak",
    "blue jay",
    "green jay",
    "rusty blackbird",
    "worm eating warbler",
    "rufous hummingbird",
    "common yellowthroat",
    "henslow sparrow",
    "cape glossy starling",
    "chuck will widow",
    "black throated sparrow",
    "vesper sparrow",
    "lincoln sparrow",
    "cardinal",
    "tropical kingbird",
    "least flycatcher",
    "yellow headed blackbird",
    "olive sided flycatcher",
    "pied kingfisher",
    "red winged blackbird",
    "spotted catbird",
    "black and white warbler",
    "nighthawk",
    "brewer blackbird",
    "white breasted nuthatch",
    "savannah sparrow",
    "hooded oriole",
    "sayornis",
    "magnolia warbler",
    "yellow warbler",
    "bobolink",
    "geococcyx",
    "tennessee warbler",
    "kentucky warbler",
    "horned lark",
]
WATER_BIRDS_LIST = [
    "pelagic cormorant",
    "laysan albatross",
    "eastern towhee",
    "white pelican",
    "ring billed gull",
    "pomarine jaeger",
    "long tailed jaeger",
    "western gull",
    "parakeet auklet",
    "artic tern",
    "least auklet",
    "herring gull",
    "common tern",
    "black footed albatross",
    "brandt cormorant",
    "eared grebe",
    "caspian tern",
    "rhinoceros auklet",
    "pacific loon",
    "elegant tern",
    "horned grebe",
    "glaucous winged gull",
    "northern fulmar",
    "crested auklet",
    "frigatebird",
    "forsters tern",
    "pigeon guillemot",
    "heermann gull",
    "sooty albatross",
    "red faced cormorant",
    "brown pelican",
    "western wood pewee",
    "pied billed grebe",
    "red breasted merganser",
    "california gull",
    "red legged kittiwake",
    "western grebe",
    "black tern",
    "horned puffin",
    "hooded merganser",
    "western meadowlark",
    "slaty backed gull",
    "mallard",
    "gadwall",
    "ivory gull",
    "least tern",
]


ALL_BIRDS_LIST_PATH = (
    "/mnt/lustre/work/oh/arubinstein17/cache/Waterbirds/all_birds.pt"
)


def make_path2label_waterbirds(dataset_path):
    return make_custom_folder_path2label(dataset_path)


def get_wb_bird_names():
    land_birds_list = LAND_BIRDS_LIST
    water_birds_list = WATER_BIRDS_LIST
    return land_birds_list, water_birds_list


def get_clip_wb_category_list():
    land_birds_list, water_birds_list = get_wb_bird_names()
    categories = land_birds_list + water_birds_list
    return [
        f"A photo of {category_name.replace('_', ' ')}"
        for category_name in categories
    ]


class WBcategories:
    def __init__(self):
        land_birds_list, water_birds_list = get_wb_bird_names()
        num_land_birds = len(land_birds_list)
        num_water_birds = len(water_birds_list)
        self.wb_to_bird_names = {
            0: list(range(num_land_birds)),
            1: list(range(num_land_birds, num_land_birds + num_water_birds)),
        }
        self.categories = ["landbird", "waterbird"]
        self.cat_to_id = {cat: i for i, cat in enumerate(self.categories)}

    def __call__(self, category_name):
        category_id = self.cat_to_id[category_name]
        return self.wb_to_bird_names[category_id]


def make_waterbirds_clip_mapper():
    mapper = make_to_classes_mapping(
        WBcategories(), aggregation_function=torch_max_func
    )
    return mapper


def make_waterbirds_clip_wrapper(model):
    return make_model_classes_wrapper(model, make_waterbirds_clip_mapper)


def make_mapping_dict_waterbirds(
    images_folder, masks_path, bboxes_path, separate_masks_folder
):
    return make_mapping_dict_generic(
        images_folder,
        masks_path,
        bboxes_path=bboxes_path,
        separate_masks_folder=separate_masks_folder,
        path2label_func=make_path2label_waterbirds,
    )
