import torch


def dinov2(repo, model_name, weight_path, source='local'):
    if source == 'local':
        model = torch.hub.load(repo, model_name, pretrained=False, source='local')
        model.load_state_dict(torch.load(weight_path), strict=True)
    else:
        model = torch.hub.load(repo, model_name)

    return model
    