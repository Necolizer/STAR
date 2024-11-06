import os
import pickle
from glob import glob
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

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

def load_pkl(pkl_file: str):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    return data

class Feeder(Dataset):
    """
    Data loader for the Harper (3D) dataset.
    This data loader is designed to provide data for forecasting, but can easily adapted as per your needs.
    """

    def __init__(self, 
                data_path: str, 
                video_path: str,
                split: str,
                p_interval=1,
                window_size=-1,
                random_rot=False, 
                entity_rearrangement=False,
                debug=False,
                img_size=224,
                uniform=False, 
                thres=64,
                rgb_frame_num=8,
            ) -> None:
        # Sanity checks
        assert os.path.exists(data_path), f"Path {data_path} does not exist. Please download the dataset first"
        assert split in ["train", "test"], f"Split {split} not recognized. Use either 'train' or 'test'"
        data_folder = os.path.join(data_path, split)
        assert os.path.exists(data_folder), f"Path {data_folder} does not exist. It is in the correct format? Refer to the README"

        self.data_path = data_path
        self.video_path = video_path
        self.split = split
        self.p_interval = p_interval
        self.window_size = window_size
        self.random_rot = random_rot
        self.entity_rearrangement = entity_rearrangement
        self.debug = debug
        self.img_size = img_size
        self.uniform = uniform
        self.thres = thres
        self.rgb_frame_num = rgb_frame_num

        # Load data
        pkls_files: list[str] = glob(os.path.join(data_folder, "*.pkl"))
        self.all_sequences: list[dict[int, dict]] = [load_pkl(f) for f in pkls_files]
        self.sample_name: list[str] = [os.path.basename(name).replace(".pkl", "") for name in pkls_files]

        self.action2label = {
            "act1": 0, "act2": 1, "act3": 2, "act4": 3, "act5": 4,
            "act6": 5, "act7": 6, "act8": 7, "act9": 8, "act10": 9,
            "act11": 10, "act12": 11, "act13": 12, "act14": 13, "act15": 14,
        }

        if self.split == 'train':
            self.video_transform = transforms.Compose([
                grouptransforms.GroupMultiScaleCrop(img_size),
                grouptransforms.GroupRandomHorizontalFlip(),
            ])
        else:
            self.video_transform = transforms.Compose([
                grouptransforms.GroupCenterCrop(img_size),
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
        self.video_list = os.listdir(self.video_path)

    def __len__(self):
        return len(self.all_sequences)

    def __pad_tensor(self, person_kpts, robot_kpts):
        # pad the person keypoints with 0
        person_kpts = F.pad(person_kpts, (0,0,0,robot_kpts.size(1)-person_kpts.size(1)), "constant", 0)

        poses = torch.stack([person_kpts, robot_kpts], dim=-1)

        return poses.permute(2,0,1,3) # T,V,C,M -> C,T,V,M
    
    def load_images_from_folder(self, folder_path):
        image_list = []
        files = os.listdir(folder_path)
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))

        # For videos in HARPER, we notice that the first about 30% frames do not include interactions or motions
        # So this sampling will skip the first about 30% frames
        idx = self.sampler.sampling(int(len(files) * 0.7))
        idx = [i + int(len(files) * 0.3) if i != -1 else -1 for i in idx]

        files = [files[i] for i in idx]

        for img_name in files:
            img_path = os.path.join(folder_path, img_name)
            image = Image.open(img_path).convert('RGB')
            image_list.append(image)
        
        return self.video_transform(image_list)

    def __getitem__(self, idx):
        curr_data = self.all_sequences[idx]
        info = self.sample_name[idx].split('_')
        subject = info[0]
        action = info[1]
        freq = info[2]
        
        human = [curr_data[i]["human_joints_3d"] for i in range(len(curr_data))]
        human = torch.tensor(human, dtype=torch.float32)
        spot = [curr_data[i]["spot_joints_3d"] for i in range(len(curr_data))]
        spot = torch.tensor(spot, dtype=torch.float32)

        poses = self.__pad_tensor(human, spot)

        poses = np.array(poses)
        valid_frame_num = np.sum(poses.sum(0).sum(-1).sum(-1) != 0)
        if self.uniform:
            poses, index_t = tools.valid_crop_uniform(poses, valid_frame_num, self.p_interval, self.window_size, self.thres)
        else:
            # poses, index_t = tools.valid_crop_resize(poses, valid_frame_num, self.p_interval,
            #                                               self.window_size, self.thres)
            poses = tools.valid_crop_resize(poses, valid_frame_num, self.p_interval, self.window_size)
        
        if self.random_rot:
            poses = tools.random_rot(poses)
        else:
            poses = torch.from_numpy(poses)
        if self.entity_rearrangement:
            poses = poses[:,:,:,torch.randperm(poses.size(3))]

        video_name_abbr = curr_data[0]["action"] + '_' + curr_data[0]["subject"]
        vname = next((i for i in self.video_list if video_name_abbr in i), '')

        img_list = self.load_images_from_folder(os.path.join(self.video_path, vname))
        img_list = [self.img_transform(i) for i in img_list]
        video = torch.stack(img_list, dim=1) # C, T, H, W

        return poses, video, self.action2label[action], idx
    
if __name__ == '__main__':
    f = Feeder(
        data_path = './dataset/HARPER/30hz',
        video_path = './dataset/HARPER/harper_rgb_FoI',
        split = 'test',
        debug = False,
        p_interval = [0.95],
        window_size = 120,
        img_size=224,
        uniform=True, 
        thres=64,
        rgb_frame_num=8,
    )

    poses, video, label, idx = f[0]
    print(poses.shape)
    print(video.shape)
    print(label)