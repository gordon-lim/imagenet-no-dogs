dependencies = ['torch']
import os
import torch
from torchvision.models.resnet import resnet50 as _resnet50

# resnet18 is the name of entrypoint
def resnet50(**kwargs):
    """ # This docstring shows up in hub.help()
    Resnet50 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    model = _resnet50(**kwargs)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 880)

    checkpoint = 'https://cumberland.isis.vanderbilt.edu/gordon/model_best.pth.tar'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False))

    return model