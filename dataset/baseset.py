from torch.utils.data import Dataset
import torch
import json
import os
import random
import time
import cv2
from PIL import Image
import torchvision.transforms as transforms
import dataset.data_transform.transforms as extended_transforms
import dataset.data_transform.modified_randaugment as rand_augment
import numpy as np


class BaseSet(Dataset):
    def __init__(self, root, mode="train", tta=None):
        self.mode = mode
        self.input_size = (224, 224)
        self.data_type = "jpg"
        self.color_space = "RGB"
        self.size = self.input_size
        self.data_root = root
        if tta is None:
            self.tta_num = 16
        else:
            self.tta_num = tta

        if self.mode == "train":
            print("Loading train data ...", end=" ")
            self.json_path = os.path.join(
                self.data_root, 'converted_ISIC_2018_Train_e.json')
        elif self.mode == "valid":
            print("Loading valid data ...", end=" ")
            self.json_path = os.path.join(
                self.data_root, 'converted_ISIC_2018_Val_e.json')
        elif self.mode == "test":
            print("Loading test data ...", end=" ")
            self.json_path = os.path.join(
                self.data_root, 'converted_ISIC_2018_Test.json')
        else:
            raise NotImplementedError

        with open(self.json_path, "r") as f:
            self.all_info = json.load(f)
        self.num_classes = self.all_info["num_classes"]
        self.data = self.all_info["annotations"]
        print("Contain {} images of {} classes".format(
            len(self.data), self.num_classes))

        self.val_sample_repeat_num = 0
        if self.mode != "train":
            self._order_crop_scale_data()

    def __getitem__(self, index):
        now_info = self.data[index]
        img = self._get_image(now_info)
        image = self.image_transform(img, index)
        image_label = (
            now_info["category_id"] if "test" not in self.mode else -1
        )
        return image, image_label

    def _order_crop_scale_data(self):
        self.val_sample_repeat_num += self.tta_num
        self.data = np.array(self.data).repeat(
            self.val_sample_repeat_num).tolist()

    def image_pre_process(self, img):
        img = Image.fromarray(img)
        # the short side of the input image resize to a fix size
        resizing = transforms.Resize(450)
        img = resizing(img)
        return img

    def image_post_process(self, img):
        # change the format of 'img' to tensor, and change the storage order from 'H x W x C' to 'C x H x W'
        # change the value range of 'img' from [0, 255] to [0.0, 1.0]
        to_tensor = transforms.ToTensor()
        img = to_tensor(img)
        normalize = transforms.Normalize(torch.from_numpy(np.array([0.485, 0.456, 0.406])),
                                         torch.from_numpy(np.array([0.229, 0.224, 0.225])))
        return normalize(img)

    def image_transform(self, img, index):
        img = self.image_pre_process(img)

        if self.mode == "train":
            img = self._train_transform(img, index)
        else:
            img = self._val_transform(img, index)

        img = self.image_post_process(img)
        return img

    def _train_transform(self, img, index):
        # need data augmentation
        # need another image
        while True:
            rand = np.random.randint(0, len(self.data))
            if rand != index:
                break
        bg_info = self.data[rand]
        img_bg = self._get_image(bg_info)
        img_bg = self.image_pre_process(img_bg)

        img = self.data_augment_train(img, img_bg)
        return img

    def data_augment_train(self, img, img_bg):
        img = torch.from_numpy(np.array(img, dtype=np.uint8))
        img_bg = torch.from_numpy(np.array(img_bg, dtype=np.uint8))
        blank_replace = tuple([i * 255.0 for i in [0.485, 0.456, 0.406]])

        # Modified RandAugment
        img = rand_augment.distort_image_with_modified_randaugment(
            img, "v3_1", img_bg, blank_replace, 0.7, 10)

        # --- random crop ---
        img = Image.fromarray(img.numpy())
        # transforms.RandomCrop(self.input_size),
        crop_method = extended_transforms.RandomCropInRate(
            nsize=self.input_size, rand_rate=(1.0, 1.0))
        img = crop_method(img)
        # img.show()
        return img

    def _val_transform(self, img, index):
        if self.val_sample_repeat_num == 0:     # simple center crop
            crop_method = transforms.CenterCrop(self.input_size)
            img = crop_method(img)
        else:
            idx = index % self.val_sample_repeat_num
            if idx < self.tta_num:   # multi crop
                img = self._val_multi_crop(img, idx)
            else:    # multi scale
                idx -= self.tta_num
                img = self._val_multi_scale(img, idx)
        # img.show()
        return img

    def _val_multi_crop(self, img, idx):
        img = torch.from_numpy(np.array(img, dtype=np.uint8))
        img_size = img.size()
        num = np.int32(np.sqrt(self.tta_num))
        y_n = int(idx / num)
        x_n = idx % num
        if img_size[1] >= img_size[0]:
            x_region = int(
                img_size[1] * 1.0)
            y_region = int(
                img_size[0] * 1.0)
        else:
            x_region = int(
                img_size[1] * 1.0)
            y_region = int(
                img_size[0] * 1.0)
        if x_region < self.input_size[1]:
            x_region = self.input_size[1]
        if y_region < self.input_size[0]:
            y_region = self.input_size[0]
        x_cut = int((img_size[1] - x_region) / 2)
        y_cut = int((img_size[0] - y_region) / 2)

        x_loc = x_cut + int(x_n * (x_region - self.input_size[1]) / (num - 1))
        y_loc = y_cut + int(y_n * (y_region - self.input_size[0]) / (num - 1))
        # Then, apply current crop
        img = img[y_loc:y_loc + self.input_size[0],
                  x_loc:x_loc + self.input_size[1], :]
        img = Image.fromarray(img.numpy())
        return img

    # def _val_multi_scale(self, img, idx):
    #     factor = float(
    #         self.cfg.TRAIN.SAMPLER.MULTI_SCALE.SCALE_NAME[idx][-3:]) / 100.0 + 1.0
    #     new_height = round(self.input_size[0] * factor)
    #     new_width = round(self.input_size[1] * factor)
    #     crop_method = transforms.CenterCrop((new_height, new_width))
    #     img = crop_method(img)
    #     img = img.resize(
    #         (self.input_size[1], self.input_size[0],), Image.ANTIALIAS)

    #     if "flip_x" in self.cfg.TRAIN.SAMPLER.MULTI_SCALE.SCALE_NAME[idx]:
    #         img = img.transpose(Image.FLIP_LEFT_RIGHT)
    #     elif "flip_y" in self.cfg.TRAIN.SAMPLER.MULTI_SCALE.SCALE_NAME[idx]:
    #         img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     elif "rotate_90" in self.cfg.TRAIN.SAMPLER.MULTI_SCALE.SCALE_NAME[idx]:
    #         img = img.transpose(Image.ROTATE_90)
    #     elif "rotate_270" in self.cfg.TRAIN.SAMPLER.MULTI_SCALE.SCALE_NAME[idx]:
    #         img = img.transpose(Image.ROTATE_270)
    #     return img

    def get_num_classes(self):
        return self.num_classes

    def get_num_class_list(self):
        class_sum = np.array([0] * self.num_classes)
        if self.mode != "test":
            for anno in self.data:
                category_id = anno["category_id"]
                class_sum[category_id] += 1
        return class_sum.tolist()

    def get_annotations(self):
        return self.data

    def get_image_id_list(self):
        image_id_list = []
        if self.val_sample_repeat_num != 0 and self.mode != "train":
            gap = self.val_sample_repeat_num
        else:
            gap = 1
        for i in range(0, len(self.data), gap):
            image_id = self.data[i]["image_id"]
            image_id_list.append(image_id)

        return image_id_list

    def get_num_images(self):
        if self.val_sample_repeat_num != 0 and self.mode != "train":
            return int(len(self.data) / self.val_sample_repeat_num)
        else:
            return len(self.data)

    def __len__(self):
        return len(self.data)

    def imread_with_retry(self, fpath):
        retry_time = 10
        for k in range(retry_time):
            try:
                img = cv2.imread(fpath)
                if img is None:
                    print("img is None, try to re-read img")
                    continue
                return img
            except Exception as e:
                if k == retry_time - 1:
                    assert False, "cv2 imread {} failed".format(fpath)
                time.sleep(0.1)

    def _get_image(self, now_info):
        if self.data_type == "jpg":
            fpath = os.path.join(self.data_root, now_info["derm_path"])
            img = self.imread_with_retry(fpath)
        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.data):
            cat_id = (
                anno["category_id"] if "category_id" in anno else anno["image_label"]
            )
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.num_classes):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i
