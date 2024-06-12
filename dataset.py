# external libs
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
import random
from os.path import join as ospj
from glob import glob
# torch libs
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
# local libs
from utils.utils import get_norm_values, chunks
from models.image_extractor import get_image_extractor
from itertools import product

import re
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ImageLoader:
    def __init__(self, root):
        self.root_dir = root

    # __call__在被调用时候才执行
    def __call__(self, img):
        '''
            If the code run in the linux,the first code below should be commented.
            If the code run in the windows environment,the environment will chang the invalid character into '_',
            the first code below will deal the problem.
            Besides,make sure the computer configuration->file system allow win32-long
            -path, because the file names in the dataset are rather long.
        '''
        # img = re.sub(r'\\|:|\*|\?|"|<|>|\|', '_', img)
        img = Image.open(ospj(self.root_dir, img)).convert('RGB')  # We don't want alpha
        return img  # img = Image.open(ospj(self.root_dir,img)).convert('RGB')


def dataset_transform(phase, norm_family='imagenet'):
    '''
        Inputs
            phase: String controlling which set of transforms to use
            norm_family: String controlling which normaliztion values to use
        
        Returns
            transform: A list of pytorch transforms
    '''
    mean, std = get_norm_values(norm_family=norm_family)

    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    elif phase == 'val' or phase == 'test':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif phase == 'all':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError('Invalid transform')

    return transform


def filter_data(all_data, pairs_gt, topk=5):
    '''
    Helper function to clean data
    '''
    valid_files = []
    with open('/home/ubuntu/workspace/top' + str(topk) + '.txt') as f:
        for line in f:
            valid_files.append(line.strip())

    data, pairs, attr, obj = [], [], [], []
    for current in all_data:
        if current[0] in valid_files:
            data.append(current)
            pairs.append((current[1], current[2]))
            attr.append(current[1])
            obj.append(current[2])

    counter = 0
    for current in pairs_gt:
        if current in pairs:
            counter += 1
    print('Matches ', counter, ' out of ', len(pairs_gt))
    print('Samples ', len(data), ' out of ', len(all_data))
    return data, sorted(list(set(pairs))), sorted(list(set(attr))), sorted(list(set(obj)))


class CompositionDataset(Dataset):
    '''
    Inputs
        root: String of base dir of dataset
        phase: String train, val, test
        split: String dataset split
        subset: Boolean if true uses a subset of train at each epoch
        num_negs: Int, numbers of negative pairs per batch
        pair_dropout: Percentage of pairs to leave in current epoch
    '''

    def __init__(
            self,
            args,
            root,
            phase,
            split='compositional-split',
            model='resnet18',
            norm_family='imagenet',
            num_negs=1,
            use_precomputed_features=False,
            return_images=False,
            train_only=False,
            open_world=False
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.args = args
        self.norm_family = norm_family
        self.return_images = return_images
        self.open_world = open_world
        self.use_precomputed_features = use_precomputed_features
        # feat_dim
        if 'resnet18' in model:
            self.feat_dim = 512
        elif 'vit-base' in model:
            self.feat_dim = 768
        elif 'vit-small' in model:
            self.feat_dim = 384
        else:
            self.feat_dim = 2048
        self.attrs, self.objs, self.pairs, self.train_pairs, self.val_pairs, self.test_pairs = self.parse_split()
        self.train_data, self.val_data, self.test_data = self.get_split_info()
        self.full_pairs = list(product(self.attrs, self.objs))

        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        if self.open_world:
            self.pairs = self.full_pairs
        self.all_pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        # 当train时只将训练集中的所有pair标签化
        if train_only and self.phase == 'train':
            print('Using only train pairs')
            self.pair2idx = {pair: idx for idx, pair in enumerate(self.train_pairs)}
        else:
            # 当val时，使用所有pair
            print('Using all dataset pairs：')
            self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        elif self.phase == 'test':
            self.data = self.test_data
        elif self.phase == 'all':
            print('Using all data')
            self.data = self.train_data + self.val_data + self.test_data
        else:
            raise ValueError('Invalid training phase')

        self.all_data = self.train_data + self.val_data + self.test_data
        print('Dataset loaded')
        print('All attrs: {}, All objs: {}, All pairs in dataset: {}'.format(
            len(self.attrs), len(self.objs), len(self.pairs)))
        print('Train pairs: {}, Validation pairs: {}, Test Pairs: {}'.format(
            len(self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('Train images: {}, Validation images: {}, Test images: {}'.format(
            len(self.train_data), len(self.val_data), len(self.test_data)))

        #  Keeping a list of all pairs that occur with each object
        self.train_obj_affordance = {}
        self.train_attr_affordance = {}
        for _obj in self.objs:
            candidates = [img for (img, attr, obj) in self.train_data if obj == _obj]
            self.train_obj_affordance[_obj] = list(candidates)
        for _attr in self.attrs:
            candidates = [img for (img, attr, obj) in self.train_data if attr == _attr]
            self.train_attr_affordance[_attr] = list(candidates)
        # Construct a list for similarity-based sample strategy
        self.sample_obj_affordance = {}
        for _obj in self.objs:
            attr_affordance = {}
            for _attr in self.attrs:
                attr_affordance[_attr] = []
            for (img, attr, obj) in self.train_data:
                if obj == _obj:
                    attr_affordance[attr].append(img)
            self.sample_obj_affordance[_obj] = attr_affordance
        self.sample_indices = list(range(len(self.data)))  # 所有test数据索引30338；val数据10240
        self.sample_pairs = self.train_pairs  # 1262

        # Load based on what to output
        self.transform = dataset_transform(self.phase, self.norm_family)  # 用于对读取的数据进行预处理
        self.loader = ImageLoader(ospj(self.root, 'images'))  # 返回数据图片的地址

        if self.use_precomputed_features:
            with torch.no_grad():
                feats_file = ospj(root, self.phase + '-' + model + '_feats_vectors.t7')
                if not os.path.exists(feats_file):
                    self.activations = self.generate_features(out_file=feats_file, model=model, args=args)
                else:
                    activation_data = torch.load(feats_file)
                    self.activations = dict(zip(activation_data['files'], activation_data['features']))

    def generate_features(self, model, args, out_file=None):
        # data = self.all_data
        files_all = []
        for item in self.data:
            files_all.append(item[0])
        transform = dataset_transform('all', self.norm_family)

        feat_extractor = get_image_extractor(arch=model).eval()
        feat_extractor = feat_extractor.to(device)
        image_feats = []
        image_files = []
        for chunk in tqdm(chunks(files_all, 256), total=len(files_all) // 256, desc=f'Extracting features {model}'):
            files = chunk
            imgs = list(map(self.loader, files))
            imgs = list(map(transform, imgs))
            feats = feat_extractor(torch.stack(imgs, 0).to(device))
            image_feats.append(feats.data.cpu())
            image_files += files
        image_feats = torch.cat(image_feats, 0)
        print('features for %d images generated' % (len(image_files)))
        activation = dict(zip(image_files, image_feats))
        torch.save({'features': image_feats, 'files': image_files}, out_file)
        return activation

    def parse_split(self):
        '''
        Helper function to read splits of object atrribute pair
        Returns
            all_attrs: List of all attributes
            all_objs: List of all objects
            all_pairs: List of all combination of attrs and objs
            tr_pairs: List of train pairs of attrs and objs
            vl_pairs: List of validation pairs of attrs and objs
            ts_pairs: List of test pairs of attrs and objs
        '''

        def parse_pairs(pair_list):

            with open(pair_list, 'r') as f:
                # 先换行符分割，然后去掉空白
                pairs = f.read().strip().split('\n')
                # 对每一个条目，用空格分割成一个二元列表
                pairs = [line.split() for line in pairs]
                # 用map将每一个二元列表转换为二元组
                pairs = list(map(tuple, pairs))
            # 利用 * 进行解压

            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            ospj(self.root, self.split, 'train_pairs.txt')
        )
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            ospj(self.root, self.split, 'val_pairs.txt')
        )
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            ospj(self.root, self.split, 'test_pairs.txt')
        )

        # now we compose all objs, attrs and pairs
        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
            list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def get_split_info(self):
        '''
        Helper method to read image, attrs, objs samples

        Returns
            train_data, val_data, test_data: List of tuple of image, attrs, obj
        '''
        data = torch.load(ospj(self.root, 'metadata_{}.t7'.format(self.split)))

        train_data, val_data, test_data = [], [], []

        for instance in data:
            image, attr, obj, settype = instance['image'], instance['attr'], \
                instance['obj'], instance['set']
            curr_data = [image, attr, obj]

            if attr == 'NA' or (attr, obj) not in self.pairs or settype == 'NA':
                # Skip incomplete pairs, unknown pairs and unknown set
                continue

            if settype == 'train':
                train_data.append(curr_data)
            elif settype == 'val':
                val_data.append(curr_data)
            else:
                test_data.append(curr_data)

        return train_data, val_data, test_data

    def get_dict_data(self, data, pairs):
        data_dict = {}
        for current in pairs:
            data_dict[current] = []

        for current in data:
            image, attr, obj = current
            data_dict[(attr, obj)].append(image)

        return data_dict


    def __getitem__(self, index):
        '''
        Call for getting samples
        '''
        index = self.sample_indices[index]

        image, attr, obj = self.data[index]  # return image_path,attr name,obj name

        # Decide what to output，when phase is train,self.finetune_backbone is true
        if self.use_precomputed_features:
            img = self.activations[image]
        else:
            img = self.loader(image)
            img = self.transform(img)
        data = [img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]]
        # Return image paths if requested as the last element of the list
        if self.return_images and self.phase != 'train':
            data.append(image)
        return data

    def __len__(self):
        '''
        Call for length
        '''
        return len(self.sample_indices)  # when train：30338
