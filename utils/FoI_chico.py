import cv2
import os
from tqdm import tqdm

raw_video_root = r'./dataset/chico/dataset_raw'
save_root = r'./dataset/chico/dataset_FoI'

for i, name in enumerate(tqdm(os.listdir(raw_video_root), ncols=80, desc='FoI')):
    video_file_path = os.path.join(raw_video_root, name, '00_03.mp4')

    assert os.path.exists(video_file_path)

    save_path = os.path.join(save_root, name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    cap = cv2.VideoCapture(video_file_path)
    frame_idx = 0
    ret = True
    while ret:
        ret, rgb_img = cap.read()  # read each frame
        if (not ret):
            break

        rgb_img = rgb_img[270:270+1090,600:600+1090,:]
        rgb_img = cv2.resize(rgb_img, (256, 256), interpolation=cv2.INTER_AREA) # resize
        save_name = os.path.join(save_path, str(frame_idx) + '.jpg')
        cv2.imwrite(save_name, rgb_img)
        frame_idx += 1
        