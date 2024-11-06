# 20241012
import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import time

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


def load_image(image_pil):
    # load image
    # image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image

def load_video(video_path):
    frame_list = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        exit(1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        frame_list.append(pil_image)
    
    return frame_list

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

def get_square_enclosing_box(boxes, W, H):
    """
    输入: 
    boxes: [N, 4] 的 tensor, 每一行是一个 bounding box (x1, y1, x2, y2)
    W: 原图片的宽度
    H: 原图片的高度
    
    输出: 
    一个 tensor (x1, y1, x2, y2) 表示最小正方形的外接框
    """
    # 分别获取所有框的最小x1和y1值，最大x2和y2值
    x1_min = torch.min(boxes[:, 0])
    y1_min = torch.min(boxes[:, 1])
    x2_max = torch.max(boxes[:, 2])
    y2_max = torch.max(boxes[:, 3])
    
    # 计算宽度和高度
    width = x2_max - x1_min
    height = y2_max - y1_min
    
    # 正方形的边长为宽度和高度中的较大者
    side_length = max(width, height)
    
    # 判断 side_length 是否符合条件
    if H < side_length:
        side_length = H

    center_x = (x1_min + x2_max) / 2
    center_y = (y1_min + y2_max) / 2
    
    # 正方形框的左上角和右下角坐标
    new_x1 = center_x - side_length / 2
    new_y1 = center_y - side_length / 2
    new_x2 = center_x + side_length / 2
    new_y2 = center_y + side_length / 2
    
    if new_x1 < 0:
        center_x = (side_length) / 2
    if new_x2 > W:
        center_x = W - (side_length) / 2
    if new_y1 < 0:
        center_y = (side_length) / 2
    if new_y2 > H:
        center_y = H - (side_length) / 2

    # 正方形框的左上角和右下角坐标
    new_x1 = center_x - side_length / 2
    new_y1 = center_y - side_length / 2
    new_x2 = center_x + side_length / 2
    new_y2 = center_y + side_length / 2

    # 确保外接框不超出图片的范围
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(W, new_x2)
    new_y2 = min(H, new_y2)
    
    return torch.tensor([new_x1, new_y1, new_x2, new_y2])

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--root_path", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--resize_shape", type=int, default=256, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    root_path = args.root_path
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.box_threshold
    device = args.device
    resize_shape = args.resize_shape

    video_list = os.listdir(root_path)
    video_list.sort()

    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    for i, v in enumerate(tqdm(video_list, ncols=80)):

        video_path = os.path.join(root_path, v)
        save_path = os.path.join(output_dir, os.path.splitext(v)[0])

        # make dir
        os.makedirs(save_path, exist_ok=True)

        frame_list = load_video(video_path)

        boxes_filt_list = []

        for i, frame in enumerate(frame_list):
            # load image
            image = load_image(frame)
            # run grounding dino model
            boxes_filt, pred_phrases = get_grounding_output(
                model, image, text_prompt, box_threshold, text_threshold, device=device
            )
            boxes_filt_list.append(boxes_filt)

        boxes_filt = torch.cat(boxes_filt_list, dim=0)

        size = frame_list[0].size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        final_box = get_square_enclosing_box(boxes_filt, W, H)

        for i, frame in enumerate(frame_list):
            resized_image = frame.crop((int(final_box[0]), int(final_box[1]), int(final_box[2]), int(final_box[3]))).resize((resize_shape, resize_shape))
            resized_image.save(os.path.join(save_path, str(i) + '.jpg'), format='JPEG')
