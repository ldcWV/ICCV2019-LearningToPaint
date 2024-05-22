import sys
import json
import torch
import numpy as np
import argparse
import torchvision.transforms as transforms
import cv2
from DRL.ddpg import decode
from utils.util import *
from PIL import Image
from torchvision import transforms, utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import glob
import os

aug = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             ])

width = 128
convas_area = width * width

img_train = []
img_test = []
train_num = 0
test_num = 0

class Paint:
    def __init__(self, batch_size, max_step):
        self.batch_size = batch_size
        self.max_step = max_step
        self.action_space = (13)
        self.observation_space = (self.batch_size, width, width, 7)
        self.test = False
        self.cnt = 0
        self.max_cnt = 0
        self.num_circles = 8
        
    def load_data(self):
        global train_num, test_num
        data_dir = '../data/ContourDrawingDataset'
        fnames = os.listdir(data_dir)
        for i, fname in enumerate(fnames):
            img_id = '%06d' % (i + 1)
            try:
                path = os.path.join(data_dir, fname)
                # img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                img = cv2.imread(path)
                img = cv2.resize(img, (width, width))
                if i > 300:                
                    train_num += 1
                    img_train.append(img)
                else:
                    test_num += 1
                    img_test.append(img)
            finally:
                if (i + 1) % 10000 == 0:                    
                    print('loaded {} images'.format(i + 1))
        print('finish loading data, {} training images, {} testing images'.format(str(train_num), str(test_num)))
        
    def pre_data(self, id, test):
        if test:
            img = img_test[id]
        else:
            img = img_train[id]
        if not test:
            img = aug(img)
        img = np.asarray(img)
        return np.transpose(img, (2, 0, 1))
    
    def reset(self, test=False, begin_num=False):
        self.cnt += 1
        self.test = test
        self.imgid = [0] * self.batch_size
        self.gt = torch.ones([self.batch_size, 3, width, width], dtype=torch.uint8).to(device) * 255
        for i in range(self.batch_size):
            if self.cnt <= self.max_cnt:
                center = torch.rand((self.num_circles, 2)).to(device) * width # num_circles x 2
                rad = width * 0.2 * (1.05 - self.cnt / self.max_cnt)
                idxs_x = torch.arange(width)
                idxs_y = torch.arange(width)
                x_coords, y_coords = torch.meshgrid(idxs_y, idxs_x, indexing='ij') # width x width
                grid_coords = torch.stack((y_coords, x_coords), dim=2).unsqueeze(2).to(device) # width x width x 1 x 2
                grid_coords = torch.tile(grid_coords, (1, 1, self.num_circles, 1)) # width x width x num_circles x 2
                dists = torch.linalg.norm(grid_coords - center, dim=3) # width x width x num_circles
                dists = torch.min(dists, dim=2).values # width x width
                self.imgid[i] = i
                self.gt[i, :, dists <= rad] = 0
            else:
                if test:
                    id = (i + begin_num)  % test_num
                else:
                    id = np.random.randint(train_num)
                self.imgid[i] = id
                self.gt[i] = torch.tensor(self.pre_data(id, test))
        self.tot_reward = ((self.gt.float() / 255) ** 2).mean(1).mean(1).mean(1)
        self.stepnum = 0
        self.canvas = torch.ones([self.batch_size, 3, width, width], dtype=torch.uint8).to(device) * 255
        self.lastdis = self.ini_dis = self.cal_dis()
        return self.observation()
    
    def observation(self):
        # canvas B * 3 * width * width
        # gt B * 3 * width * width
        # T B * 1 * width * width
        ob = []
        T = torch.ones([self.batch_size, 1, width, width], dtype=torch.uint8) * self.stepnum
        return torch.cat((self.canvas, self.gt, T.to(device)), 1) # canvas, img, T

    def cal_trans(self, s, t):
        return (s.transpose(0, 3) * t).transpose(0, 3)
    
    def step(self, action):
        self.canvas = (decode(action, self.canvas.float() / 255) * 255).byte()
        self.stepnum += 1
        ob = self.observation()
        done = (self.stepnum == self.max_step)
        reward = self.cal_reward() # np.array([0.] * self.batch_size)
        return ob.detach(), reward, np.array([done] * self.batch_size), None

    def cal_dis(self):
        return (((self.canvas.float() - self.gt.float()) / 255) ** 2).mean(1).mean(1).mean(1)
    
    def cal_reward(self):
        dis = self.cal_dis()
        reward = (self.lastdis - dis) / (self.ini_dis + 1e-8)
        self.lastdis = dis
        return to_numpy(reward)
