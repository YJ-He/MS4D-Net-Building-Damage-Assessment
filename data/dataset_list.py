import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


'''for 3 + 3 bands input data(pre+post)'''
# whu
rgb_mean = (0.485, 0.456, 0.406, 0.485, 0.456, 0.406)
rgb_std = (0.229, 0.224, 0.225, 0.229, 0.224, 0.225)

class MyDataset(Dataset):
    def __init__(self,
                 config,
                 args,
                 subset,
                 file_length=None):
        super(MyDataset, self).__init__()
        assert subset == 'train' or subset == 'val' or subset == 'test' or subset == 'train_unsup'

        self.args = args
        self.config = config
        self.root = args.input
        self.subset = subset
        self._file_length = file_length

        if self.config.nb_classes == 5:
            # xBD dataset
            self.mapping = {
                (0, 0, 0): 0,
                (0, 255, 0): 1,
                (255, 255, 0): 2,
                (255, 125, 0): 3,
                (255, 0, 0): 4,
            }
            self.class_names = ['Background', 'Intact', 'Minor', 'Major', 'Destroyed']

        # pre image
        self.data_list_pre = []
        with open(os.path.join(self.root, subset + '_image_pre.txt'), 'r') as f:
            for line in f:
                if line.strip('\n') != '':
                    self.data_list_pre.append(line.strip('\n'))

        # post image
        self.data_list_post = []
        with open(os.path.join(self.root, subset + '_image_post.txt'), 'r') as f:
            for line in f:
                if line.strip('\n') != '':
                    self.data_list_post.append(line.strip('\n'))

        if subset != 'train_unsup':
            # label
            if os.path.exists(os.path.join(self.root, subset + '_label.txt')):
                self.target_list = []
                with open(os.path.join(self.root, subset + '_label.txt'), 'r') as f:
                    for line in f:
                        if line.strip('\n') != '':
                            self.target_list.append(line.strip('\n'))
            assert len(self.data_list_pre) == len(self.data_list_post)

            if self._file_length is not None:
                self.data_list_pre, self.data_list_post, self.target_list = self._construct_new_file_list(self._file_length, is_UnsupData=False)
        else:
            if self._file_length is not None:
                self.data_list_pre, self.data_list_post,= self._construct_new_file_list(self._file_length, is_UnsupData=True)


    def _construct_new_file_list(self, length, is_UnsupData):
        """
        Construct new file list based on whether is unlabeled data or not
        """
        assert isinstance(length, int)
        files_len = len(self.data_list_pre)

        if length < files_len:
            if not is_UnsupData:
                return self.data_list_pre[:length], self.data_list_post[:length], self.target_list[:length]
            else:
                return self.data_list_pre[:length], self.data_list_post[:length]

        new_data_pre_list = self.data_list_pre * (length // files_len)
        new_data_post_list= self.data_list_post * (length // files_len)


        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_data_pre_list += [self.data_list_pre[i] for i in new_indices]
        new_data_post_list += [self.data_list_post[i] for i in new_indices]

        if not is_UnsupData:
            new_target_list = self.target_list * (length // files_len)
            new_target_list += [self.target_list[i] for i in new_indices]
            return new_data_pre_list, new_data_post_list, new_target_list
        else:
            return new_data_pre_list, new_data_post_list


    def mask_to_class(self, mask):
        """
        Encode class to number：form 0 to num_class-1
        """
        if self.config.nb_classes == 5:
            m = mask.long()
            return m
        else:
            assert self.config.nb_classes == 5
            exit(1)

    def train_transforms(self, image, mask):
        """
        Preprocessing and augmentation on training data (image and label)
        """
        in_size = self.config.input_size
        train_transform = A.Compose(
            [
                A.Resize(in_size, in_size, interpolation=cv2.INTER_NEAREST),
                A.HorizontalFlip(p=0.8),
                A.VerticalFlip(p=0.8),
                A.RandomRotate90(p=0.8),
                A.Transpose(p=0.8),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),

                A.Normalize(mean=rgb_mean, std=rgb_std),
                ToTensorV2(),
            ]
        )
        transformed = train_transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        mask = self.mask_to_class(mask)
        mask = mask.float()
        return image, mask

    def untrain_transforms(self, image, mask):
        """
        Preprocessing on val or test data (image and label)
        """
        untrain_transform = A.Compose(
            [
                A.Resize(self.config.eval_size, self.config.eval_size, interpolation=cv2.INTER_NEAREST),
                A.Normalize(mean=rgb_mean, std=rgb_std),
                ToTensorV2(),

            ]
        )
        transformed = untrain_transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        mask = self.mask_to_class(mask)
        mask = mask.float()
        return image, mask

    def untrain_transforms1(self, image):
        """
        Preprocessing on unlabeled data (image)
        """
        untrain_transform = A.Compose(
            [
                A.Resize(self.config.eval_size, self.config.eval_size),
                A.Normalize(mean=rgb_mean, std=rgb_std),
                ToTensorV2(),
            ]
        )
        transformed = untrain_transform(image=image)
        image = transformed["image"]

        return image

    def __getitem__(self, index):
        image_pre = cv2.imread(self.data_list_pre[index])
        image_pre = cv2.cvtColor(image_pre, cv2.COLOR_BGR2RGB)

        image_post = cv2.imread(self.data_list_post[index])
        image_post = cv2.cvtColor(image_post, cv2.COLOR_BGR2RGB)

        image = np.append(image_pre, image_post, axis=2).astype(np.uint8)

        if not self.args.only_prediction and self.subset != 'train_unsup':
            mask = np.array(Image.open(self.target_list[index])).astype(np.uint8)

        if self.subset == 'train':
            if not self.args.is_test:
                t_datas, t_targets = self.train_transforms(image, mask)
            else:
                t_datas, t_targets = self.untrain_transforms(image, mask)
            return t_datas, t_targets, self.data_list_post[index]
        elif self.subset =='train_unsup':
            t_datas = self.untrain_transforms1(image)
            return t_datas, self.data_list_post[index], index
        elif self.subset == 'val':
            t_datas, t_targets = self.untrain_transforms(image, mask)
            return t_datas, t_targets, self.data_list_post[index]
        elif self.subset == 'test':
            if not self.args.only_prediction:
                t_datas, t_targets = self.untrain_transforms(image, mask)
                return t_datas, t_targets, self.data_list_post[index]
            else:
                t_datas = self.untrain_transforms1(image)
                return t_datas, self.data_list_post[index]

    def __len__(self):

        return len(self.data_list_post)
