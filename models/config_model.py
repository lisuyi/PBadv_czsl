import torch
import torch.optim as optim
from models.image_extractor import get_image_extractor
from models.model import My_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def configure_model(args, dataset):
    image_extractor = None

    model = My_model(dataset, args)
    model = model.to(device)
    model_params = [param for name, param in model.named_parameters() if param.requires_grad]
    optim_params = [{'params': model_params}]

    if not args.use_precomputed_features:
        image_extractor = get_image_extractor(arch=args.image_extractor, pretrained=True)
        image_extractor = image_extractor.to(device)
        if args.finetune_backbone:
            ie_parameters = [param for name, param in image_extractor.named_parameters()]
            optim_params.append({'params': ie_parameters, 'lr': args.lrg})
    optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.wd)

    return image_extractor, model, optimizer
