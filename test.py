    #  Torch imports
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import numpy as np
from flags import DATA_FOLDER

cudnn.benchmark = True

# Python imports
import tqdm
from tqdm import tqdm
import os
from os.path import join as ospj

# Local imports
import dataset as dset
from models.common import Evaluator
from utils.utils import load_args
from models.config_model import configure_model
from flags import parser



device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # Get arguments and start logging
    epoch = 0
    args = parser.parse_args()
    logpath = args.logpath
    config = [os.path.join(logpath, _) for _ in os.listdir(logpath) if _.endswith('yml')][0]
    load_args(config, args)

    trainset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER, args.data_dir),
        phase='train',
        split=args.splitname,
        model=args.image_extractor,
        use_precomputed_features=args.use_precomputed_features,
        train_only=args.train_only,
        args=args
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )
    testset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER, args.data_dir),
        phase='test',
        split=args.splitname,
        model =args.image_extractor,
        use_precomputed_features=args.use_precomputed_features,
        args=args
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers)

    # Get model and optimizer
    image_extractor, model, optimizer = configure_model(args, trainset)
    args.extractor = image_extractor

    args.load = ospj(logpath, 'ckpt_best_auc.t7')  # e.g. logpath:
    checkpoint = torch.load(args.load)
    if image_extractor:
        try:
            image_extractor.load_state_dict(checkpoint['image_extractor'])
            image_extractor.eval()
        except:
            print('No Image extractor in checkpoint')
    model.load_state_dict(checkpoint['net'])
    model.eval()
    evaluator = Evaluator(testset, model)

    with torch.no_grad():
        test(epoch, image_extractor, model, testloader, evaluator, args)


def test(epoch, image_extractor, model, testloader, evaluator, args, print_results=True):
        if image_extractor:
            image_extractor.eval()
        model.eval()  # test
        accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []
        fc_pred_attr, fc_pred_obj = [], []
        for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
            data = [d.to(device) for d in data]

            if image_extractor:
                data[0] = image_extractor(data[0])
            _, _, _, predictions, pred_attr, pred_obj = model(data,epoch)

            attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]

            all_pred.append(predictions)
            # ---predict attr&obj
            fc_pred_attr.append(pred_attr)
            fc_pred_obj.append(pred_obj)
            # --- end ---
            all_attr_gt.append(attr_truth)
            all_obj_gt.append(obj_truth)
            all_pair_gt.append(pair_truth)

        if args.cpu_eval:
            all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt), torch.cat(all_obj_gt), torch.cat(all_pair_gt)
            fc_pred_attr = torch.cat(fc_pred_attr)  # ---predict attr&obj
            fc_pred_obj = torch.cat(fc_pred_obj)
        else:
            all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
                'cpu'), torch.cat(all_pair_gt).to('cpu')
            fc_pred_attr = torch.cat(fc_pred_attr).to('cpu')  # ---predict attr&obj
            fc_pred_obj = torch.cat(fc_pred_obj).to('cpu')

        all_pred_dict = {}
        # Gather values as dict of (attr, obj) as key and list of predictions as values
        if args.cpu_eval:
            for k in all_pred[0].keys():
                all_pred_dict[k] = torch.cat(
                    [all_pred[i][k].to('cpu') for i in range(len(all_pred))])
        else:
            for k in all_pred[0].keys():
                all_pred_dict[k] = torch.cat(
                    [all_pred[i][k] for i in range(len(all_pred))])

        # Calculate best unseen accuracy
        results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=args.bias, topk=args.topk)
        stats = evaluator.evaluate_predictions(results, fc_pred_attr, fc_pred_obj, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict,
                                               topk=args.topk)


        result = ''
        for key in ['AUC', 'best_hm', 'best_seen', 'best_unseen', 'closed_attr_match', 'closed_obj_match']:
            result = result + key.capitalize() + ': ' + str(round(stats[key], 4)) + '| '

        result = result + args.name
        if print_results:
            print(f'Results')
            print(result)
        return results


if __name__ == '__main__':
    main()
