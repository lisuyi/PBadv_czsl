import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import tqdm
from tqdm import tqdm
import os
from os.path import join as ospj
import csv
import random
import numpy as np

import datetime as dt
import dataset as dset
from models.common import Evaluator
from utils.utils import save_args, load_args
from models.config_model import configure_model
from flags import parser, DATA_FOLDER

best_auc, best_hm, best_seen, best_unseen, best_attr, best_obj, last_change = 0., 0., 0., 0., 0., 0., 0.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():

    args = parser.parse_args()
    load_args(args.config, args)
    print('Using dataset:{}'.format(args.dataset))
    now_time = dt.datetime.now().strftime('%F %T')
    logpath = os.path.join(args.cv_dir, args.name + '_' + now_time)
    os.makedirs(logpath, exist_ok=True)
    save_args(args, logpath, args.config)
    print('Temperature scale s&o:{}'.format(args.cosine_scale_so))
    print('Calibration weights:{}'.format(args.calibration_weights))
    writer = SummaryWriter(log_dir=logpath, flush_secs=30)

    trainset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER, args.data_dir),
        phase='train',
        split=args.splitname,
        model=args.image_extractor,
        use_precomputed_features=args.use_precomputed_features,
        train_only=args.train_only,
        args=args,
        open_world=args.open_world
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )

    testset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER, args.data_dir),
        phase=args.test_set,
        split=args.splitname,
        model=args.image_extractor,
        use_precomputed_features=args.use_precomputed_features,
        args=args,
        open_world=args.open_world
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers)
    # Get model and optimizer
    image_extractor, model, optimizer = configure_model(args, trainset)

    args.extractor = image_extractor

    train = train_normal

    evaluator_val = Evaluator(testset, model)
    print('Printing the model:\n', model)

    start_epoch = 0
    # Load checkpoint
    if args.load is not None:
        checkpoint = torch.load(args.load)
        if image_extractor:
            try:
                image_extractor.load_state_dict(checkpoint['image_extractor'])
                if args.freeze_features:
                    print('Freezing image extractor')
                    image_extractor.eval()
                    for param in image_extractor.parameters():
                        param.requires_grad = False
            except:
                print('No Image extractor in checkpoint')
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        print('Loaded model from ', args.load)

    for epoch in tqdm(range(start_epoch, args.max_epochs + 1), desc='Current epoch---'):
        train(epoch, image_extractor, model, trainloader, optimizer, writer)
        if epoch % args.eval_val_every == 0:
            with torch.no_grad():  # todo: might not be needed
                test(epoch, image_extractor, model, testloader, evaluator_val, writer, args, logpath)

    print('Best AUC achieved  is ', 100 * best_auc)
    print('Best HM achieved is', 100 * best_hm)
    print('Best seen achieved  is ', 100 * best_seen)
    print('Best unseen achieved is', 100 * best_unseen)
    print('Best attr is ', 100 * best_attr)
    print('Best obj is', 100 * best_obj)
    print('Last change is', 100 * last_change)
    f = open(logpath + '/result.txt', 'a+')
    f.write('\n')
    f.write('From:' + now_time + '  to:' + dt.datetime.now().strftime('%F %T'))
    f.write('\n')
    f.write('Best AUC:\t' + str(100 * best_auc))
    f.write('\n')
    f.write('Best HM:\t' + str(100 * best_hm))
    f.write('\n')
    f.write('Best Seen:\t' + str(100 * best_seen))
    f.write('\n')
    f.write('Best UnSeen:\t' + str(100 * best_unseen))
    f.write('\n')
    f.write('Best attr:\t' + str(100 * best_attr))
    f.write('\n')
    f.write('Best obj:\t' + str(100 * best_obj))
    f.write('\n')
    f.write('Last change:\t' + str(last_change))
    f.write('\n')
    f.close()


def train_normal(epoch, image_extractor, model, trainloader, optimizer, writer):
    if image_extractor:
        image_extractor.train()
    model.train()  # Let's switch to training
    pair_sum = [0]*len(model.dset.train_pairs)
    pair_t = [0]*len(model.dset.train_pairs)
    pair_acc = torch.zeros(len(model.dset.train_pairs))
    train_loss, train_attr_loss, train_obj_loss = 0.0, 0.0, 0.0
    for idx, data in tqdm(enumerate(trainloader), total=len(trainloader), desc='Training'):
        data = [d.to(device) for d in data]  # 前四个：img，attr，obj，pair真值。
        if image_extractor:
            data[0] = image_extractor(data[0])
        loss, pair_dict, _, _ = model(data, epoch)
        for index in range(len(pair_sum)):
            pair_sum[index] += len(pair_dict[0][f'class_{index}'])
            pair_t[index] += len(pair_dict[1][f'class_{index}'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss = train_loss / len(trainloader)  # len():计算batch数
    for i in range(len(pair_sum)):
        if pair_sum[i] != 0:
            pair_acc[i] = pair_t[i]/pair_sum[i]
        else:
            pair_acc[i] = 0
    if epoch >= model.args.start_syn-1:
        model.train_pair_acc = pair_acc
        # print(model.train_pair_acc)
    writer.add_scalar('Loss/train_total', train_loss, epoch)
    print('Epoch: {}| Loss: {} '.format(epoch, round(train_loss, 4)))  # 四舍五入，两位小数


def test(epoch, image_extractor, model, testloader, evaluator, writer, args, logpath):
    '''
    Runs testing for an epoch
    '''
    global best_auc, best_hm, best_seen, best_unseen, best_attr, best_obj, last_change

    def save_checkpoint(filename):
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
            'AUC': stats['AUC']
        }
        if image_extractor:
            state['image_extractor'] = image_extractor.state_dict()
        torch.save(state, os.path.join(logpath, 'ckpt_{}.t7'.format(filename)))

    if image_extractor:
        image_extractor.eval()
    model.eval()

    accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []
    fc_pred_attr, fc_pred_obj = [], []
    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
        data = [d.to(device) for d in data]  # img,attr_truth,obj_truth,pair_truth

        if image_extractor:
            data[0] = image_extractor(data[0])
        _, _, _, predictions, pred_attr, pred_obj = model(data, epoch)

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
    stats = evaluator.evaluate_predictions(results, fc_pred_attr, fc_pred_obj, all_attr_gt, all_obj_gt, all_pair_gt,
                                           all_pred_dict,
                                           topk=args.topk)
    stats['a_epoch'] = epoch
    result = ''
    # write to Tensorboard
    for key in stats:
        writer.add_scalar(key, stats[key], epoch)
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '

    result = result + args.name
    print(f'Test Epoch: {epoch}')
    print(result)
    if epoch > 0 and epoch % args.save_every == 0:
        save_checkpoint(epoch)
    if stats['AUC'] > best_auc:
        best_auc = stats['AUC']
        print('New best AUC:\t', 100 * best_auc)
        save_checkpoint('best_auc')
        last_change = epoch
    if stats['best_hm'] > best_hm:
        best_hm = stats['best_hm']
        print('New best HM:\t', 100 * best_hm)
        save_checkpoint('best_hm')
        last_change = epoch
    if stats['best_seen'] > best_seen:
        best_seen = stats['best_seen']
        print('New best seen:\t', 100 * best_seen)
        last_change = epoch
    if stats['best_unseen'] > best_unseen:
        best_unseen = stats['best_unseen']
        print('New best unseen:\t', 100 * best_unseen)
        last_change = epoch
    if stats['closed_attr_match'] > best_attr:
        best_attr = stats['closed_attr_match']
        print('New best attr:\t', 100 * best_attr)
        last_change = epoch
    if stats['closed_obj_match'] > best_obj:
        best_obj = stats['closed_obj_match']
        print('New best obj:\t', 100 * best_obj)
        last_change = epoch
    # Logs
    with open(ospj(logpath, 'logs.csv'), 'a') as f:
        w = csv.DictWriter(f, stats.keys())
        if epoch == 0:
            w.writeheader()
        w.writerow(stats)


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    try:
        set_seeds(3407)
        main()
    except KeyboardInterrupt:
        print('Best AUC achieved is ', best_auc)
        print('Best HM achieved is ', best_hm)
        print('Best seen achieved is ', best_seen)
        print('Best unseen achieved is ', best_unseen)
        print('Best attr is ', best_attr)
        print('Best obj is', best_obj)
