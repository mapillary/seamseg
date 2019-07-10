import glob
from itertools import chain
from os import path

import numpy as np
import torch.utils.data as data
import umsgpack
from PIL import Image


class ISSDataset(data.Dataset):
    """Instance segmentation dataset

    This assumes the dataset to be formatted as defined in:
        https://github.com/mapillary/seamseg/wiki/Dataset-format

    Parameters
    ----------
    root_dir : str
        Path to the root directory of the dataset
    split_name : str
        Name of the split to load: this must correspond to one of the files in `root_dir/lst`
    transform : callable
        Transformer function applied to the loaded entries to prepare them for pytorch. This should be callable as
        `transform(img, msk, cat, cls)`, where:
            - `img` is a PIL.Image with `mode="RGB"`, containing the RGB data
            - `msk` is a list of PIL.Image with `mode="L"`, containing the instance segmentation masks
            - `cat` is a list containing the instance id to class id mapping
            - `cls` is an integer specifying a requested class for class-uniform sampling, or None

    """
    _IMG_DIR = "img"
    _MSK_DIR = "msk"
    _LST_DIR = "lst"
    _METADATA_FILE = "metadata.bin"

    def __init__(self, root_dir, split_name, transform):
        super(ISSDataset, self).__init__()
        self.root_dir = root_dir
        self.split_name = split_name
        self.transform = transform

        # Folders
        self._img_dir = path.join(root_dir, ISSDataset._IMG_DIR)
        self._msk_dir = path.join(root_dir, ISSDataset._MSK_DIR)
        self._lst_dir = path.join(root_dir, ISSDataset._LST_DIR)
        for d in self._img_dir, self._msk_dir, self._lst_dir:
            if not path.isdir(d):
                raise IOError("Dataset sub-folder {} does not exist".format(d))

        # Load meta-data and split
        self._meta, self._images = self._load_split()

    def _load_split(self):
        with open(path.join(self.root_dir, ISSDataset._METADATA_FILE), "rb") as fid:
            metadata = umsgpack.unpack(fid, encoding="utf-8")

        with open(path.join(self._lst_dir, self.split_name + ".txt"), "r") as fid:
            lst = fid.readlines()
        lst = set(line.strip() for line in lst)

        meta = metadata["meta"]
        images = [img_desc for img_desc in metadata["images"] if img_desc["id"] in lst]

        return meta, images

    def _load_item(self, item):
        img_desc = self._images[item]

        img_file = path.join(self._img_dir, img_desc["id"])
        if path.exists(img_file + ".png"):
            img_file = img_file + ".png"
        elif path.exists(img_file + ".jpg"):
            img_file = img_file + ".jpg"
        else:
            raise IOError("Cannot find any image for id {} in {}".format(img_desc["id"], self._img_dir))
        img = Image.open(img_file).convert(mode="RGB")

        # Load all masks
        msk_file = path.join(self._msk_dir, img_desc["id"] + ".png")
        msk = [Image.open(msk_file)]
        i = 1
        while path.exists("{}.{}".format(msk_file, i)):
            msk.append(Image.open("{}.{}".format(msk_file, i)))
            i += 1

        cat = img_desc["cat"]
        iscrowd = img_desc["iscrowd"]
        return img, msk, cat, iscrowd, img_desc["id"]

    @property
    def categories(self):
        """Category names"""
        return self._meta["categories"]

    @property
    def num_categories(self):
        """Number of categories"""
        return len(self.categories)

    @property
    def num_stuff(self):
        """Number of "stuff" categories"""
        return self._meta["num_stuff"]

    @property
    def num_thing(self):
        """Number of "thing" categories"""
        return self.num_categories - self.num_stuff

    @property
    def original_ids(self):
        """Original class id of each category"""
        return self._meta["original_ids"]

    @property
    def palette(self):
        """Default palette to be used when color-coding semantic labels"""
        return np.array(self._meta["palette"], dtype=np.uint8)

    @property
    def img_sizes(self):
        """Size of each image of the dataset"""
        return [img_desc["size"] for img_desc in self._images]

    @property
    def img_categories(self):
        """Categories present in each image of the dataset"""
        return [img_desc["cat"] for img_desc in self._images]

    def __len__(self):
        return len(self._images)

    def __getitem__(self, item):
        img, msk, cat, iscrowd, idx = self._load_item(item)
        rec = self.transform(img, msk, cat, iscrowd)
        size = (img.size[1], img.size[0])

        img.close()
        for m in msk:
            m.close()

        rec["idx"] = idx
        rec["size"] = size
        return rec

    def get_raw_image(self, idx):
        """Load a single, unmodified image with given id from the dataset"""
        img_file = path.join(self._img_dir, idx)
        if path.exists(img_file + ".png"):
            img_file = img_file + ".png"
        elif path.exists(img_file + ".jpg"):
            img_file = img_file + ".jpg"
        else:
            raise IOError("Cannot find any image for id {} in {}".format(idx, self._img_dir))

        return Image.open(img_file)

    def get_image_desc(self, idx):
        """Look up an image descriptor given the id"""
        matching = [img_desc for img_desc in self._images if img_desc["id"] == idx]
        if len(matching) == 1:
            return matching[0]
        else:
            raise ValueError("No image found with id %s" % idx)


class ISSTestDataset(data.Dataset):
    _EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]

    def __init__(self, in_dir, transform):
        super(ISSTestDataset, self).__init__()
        self.in_dir = in_dir
        self.transform = transform

        # Find all images
        self._images = []
        for img_path in chain(
                *(glob.iglob(path.join(self.in_dir, '**', ext), recursive=True) for ext in ISSTestDataset._EXTENSIONS)):
            _, name_with_ext = path.split(img_path)
            idx, _ = path.splitext(name_with_ext)

            with Image.open(img_path) as img_raw:
                size = (img_raw.size[1], img_raw.size[0])

            self._images.append({
                "idx": idx,
                "path": img_path,
                "size": size,
            })

    @property
    def img_sizes(self):
        """Size of each image of the dataset"""
        return [img_desc["size"] for img_desc in self._images]

    def __len__(self):
        return len(self._images)

    def __getitem__(self, item):
        # Load image
        with Image.open(self._images[item]["path"]) as img_raw:
            size = (img_raw.size[1], img_raw.size[0])
            img = self.transform(img_raw.convert(mode="RGB"))

        return {
            "img": img,
            "idx": self._images[item]["idx"],
            "size": size,
            "abs_path": self._images[item]["path"],
            "rel_path": path.relpath(self._images[item]["path"], self.in_dir),
        }
