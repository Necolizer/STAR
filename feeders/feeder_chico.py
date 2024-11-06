import os
import glob
import pickle
from typing import Any, List, Optional, Tuple
import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import random
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

class Feeder(Dataset):
    """CHICO Dataset Dataloader

    Expecting this folder structure:

    ROOT/
        poses/
            S00/
                lift.pkl
                span_light.pkl
                place-lp_CRASH.pkl
                ...
            S01/
                lift.pkl
                span_light.pkl
                place-lp_CRASH.pkl
                ...
            ...
        rgb/
            S00/
                00_03.mp4
                00_06.mp4
                00_12.mp4
            S01/
                00_03.mp4
                00_06.mp4
                00_12.mp4
            ...

    # ----------------------------------------------------------
    Pickles of poses contains a List of time instants.
    For each time instant you will find
        - Person Keypoints 3D
        - Robot Keypoints 3D

    # ----------------------------------------------------------
    Keypoints linkings
    [[0,1], [1,2], [2,3], [0,4], [4,5], [5,6], [1,9], [4,12], [8,7], [8,9], [8,12], [9,10], [10,11], [12,13], [13,14]]

    """

    actions = [
        "hammer",
        "lift",
        "place-hp",
        "place-hp_CRASH",
        "place-lp",
        "place-lp_CRASH",
        "polish",
        "polish_CRASH",
        "span_heavy",
        "span_heavy_CRASH",
        "span_light",
        "span_light_CRASH",
    ]

    keypoints_dict = {
        "hip": 0,
        "r_hip": 1,
        "r_knee": 2,
        "r_foot": 3,
        "l_hip": 4,
        "l_knee": 5,
        "l_foot": 6,
        "nose": 7,
        "c_shoulder": 8,
        "r_shoulder": 9,
        "r_elbow": 10,
        "r_wrist": 11,
        "l_shoulder": 12,
        "l_elbow": 13,
        "l_wrist": 14,
    }

    keypoints_links = [
        [0,1],
        [1,2],
        [2,3],
        [0,4],
        [4,5],
        [5,6],
        [1,9],
        [4,12],
        [8,7],
        [8,9],
        [8,12],
        [9,10],
        [10,11],
        [12,13],
        [13,14]
    ]

    kuka_links = [
        [0,1],
        [1,2],
        [2,3],
        [3,4],
        [4,5],
        [5,6],
        [6,7],
        [7,8]
    ]

    def __init__(
        self,
        root: str,
        split: str,
        task: str = 'IR',
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
        super().__init__()

        self.root = root
        self.split = split
        self.task = task
        self.segment_length = 300
        self.stride = 50
        self.p_interval = p_interval
        self.window_size = window_size
        self.random_rot = random_rot
        self.entity_rearrangement = entity_rearrangement
        self.debug = debug
        self.img_size = img_size
        self.uniform = uniform
        self.thres = thres
        self.rgb_frame_num = rgb_frame_num

        assert os.path.isdir(root), f"Folder not found {root}!"

        poses_path = os.path.join(root, "poses")
        rgb_path = os.path.join(root, "dataset_FoI")
        self.video_path = rgb_path

        poses_found = os.path.isdir(poses_path) and len(os.listdir(poses_path)) > 1
        rgb_found = os.path.isdir(rgb_path) and len(os.listdir(rgb_path)) > 1

        assert poses_found or rgb_found, "Excpected at least Poses or RGB to be found!"

        if not poses_found:
            print(f"No Poses found in {root}")
        if not rgb_found:
            print(f"No RGB found in {root}")

        poses_pkls = glob.glob(poses_path + "/**/*.pkl", recursive=True)

        if split == 'train':
            subject_filter = ["S01", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17"]
            poses_pkls = [
                p for p in poses_pkls if os.path.basename(os.path.dirname(p)) in subject_filter
            ]
        elif split == 'val':
            subject_filter = ["S00", "S04"]
            poses_pkls = [
                p for p in poses_pkls if os.path.basename(os.path.dirname(p)) in subject_filter
            ]
        elif split == 'test':
            subject_filter = ["S02", "S03", "S18", "S19"]
            poses_pkls = [
                p for p in poses_pkls if os.path.basename(os.path.dirname(p)) in subject_filter
            ]
        else:
            NotImplementedError("Invalid Split Name")

        if task == 'IR':
            # Interaction Recognition
            self.action2label = {
                "hammer": 0,
                "lift": 1,
                "place-hp": 2,
                "place-hp_CRASH": 2,
                "place-lp": 3,
                "place-lp_CRASH": 3,
                "polish": 4,
                "polish_CRASH": 4,
                "span_heavy": 5,
                "span_heavy_CRASH": 5,
                "span_light": 6,
                "span_light_CRASH": 6,
            }
        elif task == 'UCD':
            # Unexpected Collision Detection
            self.action2label = {
                "hammer": 0,
                "lift": 0,
                "place-hp": 0,
                "place-hp_CRASH": 1,
                "place-lp": 0,
                "place-lp_CRASH": 1,
                "polish": 0,
                "polish_CRASH": 1,
                "span_heavy": 0,
                "span_heavy_CRASH": 1,
                "span_light": 0,
                "span_light_CRASH": 1,
            }
        else:
            NotImplementedError("Invalid Task Name")

        self.poses_pkls = poses_pkls
        tmp = [self.read_pickle(p) for p in poses_pkls]

        # subject, action, poses
        self.data = [
            self.__get_subject_and_action(p) + (self.__pad_list2tensor(tmp[i][0], tmp[i][1]),)
            for (i, p) in enumerate(poses_pkls)
        ]

        print(f"Found {len(self.data)} files")

        samples = []
        names = []
        clip_indices = []
        for (subject, action, poses) in self.data:
            pose_clips = self.split_tensor_on_T(poses, self.segment_length, self.stride)
            for i, clip in enumerate(pose_clips):
                samples.append((subject, action, clip))
                names.append(subject+'_'+action)
                clip_indices.append(i)
        self.data = samples
        self.sample_name = names
        self.clip_indices = clip_indices

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

    def split_tensor_on_T(self, tensor, segment_length=300, stride=50):
        T = tensor.size(1)
        
        # find the nearest multiple of 300
        T_prime = (T + segment_length - 1) // segment_length * segment_length
        
        if T < T_prime:
            # pad with 0
            padding = T_prime - T
            padding_tensor = torch.zeros((tensor.size(0), padding, tensor.size(2), tensor.size(3)))
            tensor = torch.cat([tensor, padding_tensor], dim=1)
        elif T > T_prime:
            # truncation
            tensor = tensor[:, :T_prime, :, :]
        
        # split the tensor into many tensors in a non-overlapping manner
        # split_tensors = torch.split(tensor, segment_length, dim=1)

        # split the tensor into many tensors with segment len 300 and stride 50
        split_tensors = []
        for start in range(0, tensor.size(1)-segment_length+1, stride):
            split_tensors.append(tensor[:, start : start + segment_length, :, :])
        
        return split_tensors

    def __pad_list2tensor(self, person_kpts, robot_kpts):
        # List to Tensor
        person_kpts = torch.tensor(person_kpts) # [T,15,3]
        robot_kpts = torch.tensor(robot_kpts) # [T,9,3]

        # pad the robot keypoints with 0
        robot_kpts = F.pad(robot_kpts, (0,0,0,person_kpts.size(1)-robot_kpts.size(1)), "constant", 0)

        poses = torch.stack([person_kpts, robot_kpts], dim=-1)

        return poses.permute(2,0,1,3) # T,V,C,M -> C,T,V,M

    def __get_subject_and_action(self, path: str):
        res, action = os.path.split(path)
        _, subject = os.path.split(res)

        action = action.replace(".pkl", "")

        return subject, action

    def read_pickle(self, pickle_path: str):
        person_kpts = []
        robot_kpts = []
        with open(pickle_path, "rb") as fp:
            data = pickle.load(fp)
            for d in data:
                person_kpts.append(d[0])
                robot_kpts.append(d[1])

        return person_kpts, robot_kpts

    def __len__(self):
        return len(self.data)
    
    def load_images_from_folder(self, folder_path, clip_index):
        image_list = []
        files = os.listdir(folder_path)
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))

        files = files[clip_index * self.stride : clip_index * self.stride + self.segment_length - 1]

        idx = self.sampler.sampling(len(files))
        files = [files[i] for i in idx]

        for img_name in files:
            img_path = os.path.join(folder_path, img_name)
            image = Image.open(img_path).convert('RGB')
            image_list.append(image)
        
        return self.video_transform(image_list)

    def __getitem__(
        self, index
    ) -> Tuple[torch.Tensor, int]:
        """Get item of dataset

        Args:
            index (int): index of poses and rgbs

        Returns:
        """

        subject, action, poses = self.data[index]

        poses = poses/1000 # milimeter -> meter
        
        poses = np.array(poses)
        valid_frame_num = np.sum(poses.sum(0).sum(-1).sum(-1) != 0)
        if self.uniform:
            poses, index_t = tools.valid_crop_uniform(poses, valid_frame_num, self.p_interval, self.window_size, self.thres)
        else:
            poses, index_t = tools.valid_crop_resize(poses, valid_frame_num, self.p_interval,
                                                          self.window_size, self.thres)

        if self.random_rot:
            poses = tools.random_rot(poses)
        else:
            poses = torch.from_numpy(poses)
        if self.entity_rearrangement:
            poses = poses[:,:,:,torch.randperm(poses.size(3))]

        name = self.sample_name[index]
        clip_index = self.clip_indices[index]
        img_list = self.load_images_from_folder(os.path.join(self.video_path, name), clip_index)
        img_list = [self.img_transform(i) for i in img_list]
        video = torch.stack(img_list, dim=1) # C, T, H, W

        return poses, video, self.action2label[action], index