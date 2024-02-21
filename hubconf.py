dependencies = ['torch']
import os
import torch
from collections import OrderedDict
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

    checkpoint = 
'https://cumberland.isis.vanderbilt.edu/gordon/Weights/imagenet-no-dogs/resnet50-imagenet-no-dogs.pth.tar'

    # original saved file with DataParallel
    checkpoint = torch.hub.load_state_dict_from_url(checkpoint, progress=False)
    state_dict = checkpoint['state_dict']
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    return model
