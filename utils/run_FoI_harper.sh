export CUDA_VISIBLE_DEVICES=0
python FoI_harper.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --root_path ./harper/External_RGB \
  --output_dir ./harper/harper_rgb_FoI \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --resize_shape 256 \
  --text_prompt "person . robot dog" \
  --device "cuda"