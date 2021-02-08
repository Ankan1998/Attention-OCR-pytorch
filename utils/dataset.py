import os
import random
from PIL import Image, ImageOps
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler

from utils.tokenizer import Tokenizer


def file_reader(directory, filename, sep=' '):
    examples = []
    with open(os.path.join(directory, filename), 'r') as fp:
        for line in fp.readlines():
            line=line.strip("\n")
            if sep in line:
                # txt=[]
                # image_file = line.split(sep=sep, maxsplit=-1)[0] # Change for getting multiple label
                # txt.extend(line.split(sep=sep)[1:])
                image_file,txt= line.split(sep=sep, maxsplit=1)
                image_file = os.path.abspath(os.path.join(directory, image_file))
                txt = txt.strip()
                if os.path.isfile(image_file):
                    examples.append((txt, image_file))
    random.shuffle(examples)
    return examples  # list of tuple




class Image_OCR_Dataset(Dataset):
    def __init__(self, data_dir, filename, img_width, img_height, data_file_separator,max_len=20, chars=None):
        self.list_of_tuple = file_reader(data_dir, filename, sep=data_file_separator)  # list of tuple containing (label,image_file_path)
        print(len(self.list_of_tuple))


        self.img_trans = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=(0.229, 0.224, 0.225)),
        ])

        self.max_len = max_len

        if chars is None:
            self.chars = list('1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        else:
            self.chars = list(chars)

        self.tokenizer = Tokenizer(self.chars)

        # self.first_run = True

    def __len__(self):
        return len(self.list_of_tuple)

    def __getitem__(self, idx):

        s = self.list_of_tuple[idx][0]
        d = Image.open(self.list_of_tuple[idx][1])

        label = torch.full((self.max_len + 2,), self.tokenizer.EOS_token, dtype=torch.long)

        ts = self.tokenizer.tokenize(s)
        #ts_shape = ts.shape
        label[:ts.shape[0]] = torch.tensor(ts)

        return self.img_trans(d), label

if __name__ == "__main__":

    # img_width = 160
    # img_height = 60
    # max_len = 4
    #
    # nh = 512
    #
    # teacher_forcing_ratio = 0.5
    #
    # batch_size = 4
    #
    # lr = 3e-4
    # n_epoch = 10
    #
    # n_works = 1
    # save_checkpoint_every = 5
    #
    # data_dir = "C:/Users/Ankan/Desktop/pytorch_aocr/main_dir"
    # train_file = "train.txt"
    # test_file = "test.txt"
    #
    # max_len = 4
    #
    # ds_train = Image_OCR_Dataset(data_dir, train_file, img_width, img_height, 4, max_len)
    # print(ds_train.__len__())
    # tokenizer = ds_train.tokenizer
    #
    # train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=n_works)
    # print(len(train_loader.dataset))

    zeta=file_reader(r"C:\Users\Ankan\Desktop\pytorch_aocr\main_dir", "train.txt", sep=',')
    print(zeta)