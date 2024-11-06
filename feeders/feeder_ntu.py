import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from feeders import tools
from feeders import grouptransforms


class Sampling(object):
    def __init__(self, num, interval=[0.75, 1.0]):
        assert num > 0, "at least sampling 1 frame"
        self.num = num
        self.interval = interval if type(interval) == list else [interval]

    def sampling(self, range_max):
        assert range_max > 0, \
            ValueError("range_max = {}".format(range_max))
        
        if len(self.interval) == 1:
            p = self.interval[0]
            bias = int((1-p) * range_max/2)
            total = range_max - 2*bias
            interval = max(int(total / (self.num-1)), 1)
            clip = list(range(bias,range_max-bias,interval))
        else:
            p = np.random.rand(1)*(self.interval[1]-self.interval[0])+self.interval[0]
            cropped_length = np.minimum(np.maximum(int(np.floor(range_max*p)),32), range_max)
            
            bias = np.random.randint(0,range_max-cropped_length+1)
            total = range_max - 2*bias
            interval = max(int(total / (self.num-1)), 1)
            
            clip = list(range(bias,range_max-bias,interval))

        if len(clip) > self.num:
            clip = clip[:self.num]
        elif len(clip) < self.num:
            clip = clip + [-1] * (self.num-len(clip))

        return clip #index list

class Feeder(Dataset):
    def __init__(self, data_path, video_path, sample_path, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=True,
                 bone=False, vel=False, entity_rearrangement=False, img_size=224, aug_method='z', intra_p=0.5, inter_p=0.0, uniform=False, thres=64,
                 onlyone=False, rgb_frame_num=8, video2image=None):
        """
        data_path:
        label_path:
        split: training set or test set
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move:
        random_rot: rotate skeleton around xyz axis
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
        use_mmap: If true, use mmap mode to load data, which can save the running memory
        bone: use bone modality or not
        vel: use motion modality or not
        entity_rearrangement: If true, use entity rearrangement (interactive actions)
        """

        self.debug = debug
        self.data_path = data_path
        self.video_path = video_path
        self.sample_path = sample_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.entity_rearrangement = entity_rearrangement
        self.img_size = img_size
        self.aug_method = aug_method
        self.intra_p = intra_p
        self.inter_p = inter_p
        self.uniform = uniform
        self.thres = thres
        self.rgb_frame_num = rgb_frame_num
        self.onlyone = onlyone
        self.video2image = video2image
        self.load_data()
        if normalization:
            self.get_mean_map()


        if img_size == 224:
            if self.split == 'train':
                self.video_transform = transforms.Compose([
                    grouptransforms.GroupMultiScaleCrop(224),
                    grouptransforms.GroupRandomHorizontalFlip(),
                ])
            else:
                self.video_transform = transforms.Compose([
                    grouptransforms.GroupCenterCrop(224),
                ])
        elif img_size == 256:
            if self.split == 'train':
                self.video_transform = transforms.Compose([
                    grouptransforms.GroupMultiScaleCrop(256),
                    grouptransforms.GroupRandomHorizontalFlip(),
                ])
            else:
                self.video_transform = transforms.Compose([
                    grouptransforms.GroupCenterCrop(256),
                ])
        else:
            if self.split == 'train':
                self.video_transform = transforms.Compose([
                    grouptransforms.GroupMultiScaleCrop(224),
                    grouptransforms.GroupRandomHorizontalFlip(),
                    grouptransforms.GroupScale(img_size),
                ])
            else:
                self.video_transform = transforms.Compose([
                    grouptransforms.GroupCenterCrop(224),
                    grouptransforms.GroupScale(img_size),
                ])
        
        if self.split == 'train':
            self.img_transform = transforms.Compose([
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.sampler = Sampling(num=rgb_frame_num, interval=p_interval)

    def load_data(self):
        # data: N C V T M
        if self.use_mmap:
            npz_data = np.load(self.data_path, mmap_mode='r')
        else:
            npz_data = np.load(self.data_path)

        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            
            # Mutual Actions
            filtered_index = np.where(((self.label > 48) & (self.label < 60)) | (self.label > 104))
            self.label = self.label[filtered_index]
            self.data = self.data[filtered_index]
            self.label = np.where(((self.label > 48) & (self.label < 60)), self.label-49, self.label)
            self.label = np.where((self.label > 104), self.label-94, self.label)

            self.sample_name = np.loadtxt(self.sample_path, dtype = str)
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]

            # Mutual Actions
            filtered_index = np.where(((self.label > 48) & (self.label < 60)) | (self.label > 104))
            self.label = self.label[filtered_index]
            self.data = self.data[filtered_index]
            self.label = np.where(((self.label > 48) & (self.label < 60)), self.label-49, self.label)
            self.label = np.where((self.label > 104), self.label-94, self.label)

            self.sample_name = np.loadtxt(self.sample_path, dtype = str)
        else:
            raise NotImplementedError('data split only supports train/test')

        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def load_images_from_folder(self, folder_path):
        image_list = []
        files = os.listdir(folder_path)
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))

        idx = self.sampler.sampling(len(files))
        files = [files[i] for i in idx]

        if self.onlyone:
            files = [files[len(files)//2]] # FIXME

        for img_name in files:
            img_path = os.path.join(folder_path, img_name)
            image = Image.open(img_path).convert('RGB')
            image_list.append(image)
        
        return self.video_transform(image_list)
    
    def video_2_image(self, image_list):
        num = self.video2image // self.img_size
        final_image = Image.new('RGB', (self.video2image, self.video2image))
        for i, img in enumerate(image_list):
            row = i // num  # row
            col = i % num   # col
            # paste
            position = (col * self.img_size, row * self.img_size)
            final_image.paste(img, position)

        return [final_image]

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        if self.uniform:
            data_numpy, index_t = tools.valid_crop_uniform(data_numpy, valid_frame_num, self.p_interval, self.window_size, self.thres)
        else:
            data_numpy, index_t = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval,
                                                          self.window_size, self.thres)

        if self.split == 'train':
            # intra-instance augmentation

            if self.entity_rearrangement:
                # Generate a random permutation of indices for the last dimension
                permuted_indices = np.random.permutation(data_numpy.shape[-1])
                # Apply the permutation to the last dimension of the array
                data_numpy = data_numpy[:, :, :, permuted_indices]

            p = np.random.rand(1)
            if p < self.intra_p:
                if '1' in self.aug_method:
                    data_numpy = tools.shear(data_numpy, p=0.5)
                if '2' in self.aug_method:
                    data_numpy = tools.rotate(data_numpy, p=0.5)
                if '3' in self.aug_method:
                    data_numpy = tools.scale(data_numpy, p=0.5)
                if '4' in self.aug_method:
                    data_numpy = tools.spatial_flip(data_numpy, p=0.5)
                if '8' in self.aug_method:
                    data_numpy = tools.drop_axis(data_numpy, p=0.5)
                if '9' in self.aug_method:
                    data_numpy = tools.drop_joint(data_numpy, p=0.5)

            # inter-instance augmentation
            elif (p < (self.intra_p + self.inter_p)) & (p >= self.intra_p):
                adain_idx = random.choice(np.where(self.label == label)[0])
                data_adain = self.data[adain_idx]
                data_adain = np.array(data_adain)
                f_num = np.sum(data_adain.sum(0).sum(-1).sum(-1) != 0)
                t_idx = np.round((index_t + 1) * f_num / 2).astype(int)
                data_adain = data_adain[:, t_idx]
                data_numpy = tools.skeleton_adain_bone_length(data_numpy, data_adain)

            else:
                data_numpy = data_numpy.copy()

        name = self.sample_name[index]

        img_list = self.load_images_from_folder(os.path.join(self.video_path, name))

        if self.video2image:
            img_list = self.video_2_image(img_list)

        img_list = [self.img_transform(i) for i in img_list]
        video = torch.stack(img_list, dim=1) # C, T, H, W
        
        return data_numpy, video, label, index
    
if __name__ == '__main__':
    np.random.seed(0)
    sampler = Sampling(num=8, interval=[0.5, 1.0])
    idx = sampler.sampling(80)
    print(idx)
    sampler = Sampling(num=8, interval=[0.95])
    idx = sampler.sampling(80)
    print(idx)