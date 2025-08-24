# STAR: Skeletal Token Alignment and Rearrangement for Interaction Recognition

Here's the official implementation of **STAR: Skeletal Token Alignment and Rearrangement for Interaction Recognition** accepted for publication in IEEE Transactions on Multimedia in 2025.

## News
- [2025-08] STAR has been accepted for publication in IEEE Transactions on Multimedia.

## Prerequisites
To clone the `main` branch, use the following `git` command:
```shell
git clone -b main https://github.com/Necolizer/STAR.git
```

```shell
pip install -r requirements.txt 
```

## Datasets
### Chico
Download the 3D skeleton data and the external RGB videos from [this link](https://univr-my.sharepoint.com/:f:/g/personal/federico_cunico_univr_it/Eh3Mau4d7WpLpP06TsMimzABKD344Bmy3xFFk473QlPrhA?e=rwLhhV). Utils are provided in their repo [Chico](https://github.com/AlessioSam/CHICO-PoseForecasting).

### HARPER
Download the 30Hz 3D skeleton data and the external RGB videos from [this link](https://univr-my.sharepoint.com/:f:/g/personal/federico_cunico_univr_it/Esk9qR4fKyFBg05UdXK0YSYBY8JvLHpY2Bis2xyX1pcVWg). You could download the 3D skeleton data using the script provided in [HARPER](https://github.com/intelligolabs/HARPER):
```python
PYTHONPATH=. python download/harper_only_3d_downloader.py --dst_folder ./data
```

This will generate the following tree structure:
```
data
├── harper_3d_120
│   ├── test
│   │   ├── subj_act_120hz.pkl
│   │   ├── ...
│   │   └── subj_act_120hz.pkl
│   └── train
│       ├── subj_act_120hz.pkl
│       ├── ...
│       └── subj_act_120hz.pkl
└── harper_3d_30
    ├── test
    │   ├── subj_act_30hz.pkl
    │   ├── ...
    │   └── subj_act_30hz.pkl
    └── train
        ├── subj_act_30hz.pkl
        ├── ...
        └── subj_act_30hz.pkl
```

### NTU Mutual 11 & 26
NTU Mutual 11 & 26 are subsets of the NTU RGB+D and the NTU RGB+D 120 dataset, specifically designed for interaction recognition tasks.

**DownLoad**
1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
    1. nturgbd_skeletons_s001_to_s017.zip (NTU RGB+D 60)
    2. nturgbd_skeletons_s018_to_s032.zip (NTU RGB+D 120)
    3. Extract above files to ./data/nturgbd_raw
3. Download the RGB videos

**Directory Structure**

Put downloaded data into the following directory structure:
```
- data/
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
    - rgb/
      ...
```

**Generating Data**
```shell
cd ./data/ntu120 # or cd ./data/ntu
# Get skeleton of each performer
python get_raw_skes_data.py
# Remove the bad skeleton 
python get_raw_denoised_data.py
# Transform the skeleton to the center of the first frame
python seq_transformation.py
```

## Run the Code

First preprocess the RGB video frames using `utils/FoI_[benchmarkName].py` to get FoI regions. Then specify your path in the configurations.

Run the following command to start training and evaluation:
```shell
python main.py --config config/[benchmarkName]/[yourSetting].yaml
```

## Checkpoints
Please stay tuned for updates.

## Citation

If you find this work or code helpful in your research, please consider citing:
```
Please stay tuned for updates.
```

## Acknowledgement
This project is built on top of the follows, please consider citing them if you find them useful:
- [CHASE](https://github.com/Necolizer/CHASE)
- [ISTA-Net](https://github.com/Necolizer/ISTA-Net)
- [Chico](https://github.com/AlessioSam/CHICO-PoseForecasting)
- [HARPER](https://github.com/intelligolabs/HARPER)
- [NTU RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition)