import torch
from model.Vision.VideoMAEv2.model import vit_base_patch16_224


def videomaev2(weight_path):
    model = vit_base_patch16_224(num_classes=710)
    state_dict = torch.load(weight_path, map_location='cpu')
    state_dict = state_dict['module']
    model.load_state_dict(state_dict, strict=True)
    model.reset_classifier(num_classes=0)
    return model