import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .common import MLP, found_affinity_unseen_paris, fgsm_attack, consistency_loss
from .word_embedding import load_word_embeddings
from os.path import join as ospj
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epsilon_list = [0.0, 0.005, 0.05, 0.5]


def get_all_ids(relevant_pairs, attr2idx, obj2idx):
    # Precompute validation pairs
    attrs, objs = zip(*relevant_pairs)
    attrs = [attr2idx[attr] for attr in attrs]
    objs = [obj2idx[obj] for obj in objs]
    pairs = [a for a in range(len(relevant_pairs))]
    pairs = torch.LongTensor(pairs).to(device)
    attrs = torch.LongTensor(attrs).to(device)
    objs = torch.LongTensor(objs).to(device)

    return attrs, objs, pairs


def get_word_dim(args):
    if args.emb_init == 'glove' or args.emb_init == 'word2vec' or args.emb_init == 'fasttext':
        word_dim = 300
    elif args.emb_init == 'ft+w2v+gl':
        word_dim = 900
    else:
        word_dim = 600
    return word_dim


class My_model(nn.Module):
    def __init__(self, dset, args):
        super(My_model, self).__init__()
        self.args = args
        self.dset = dset
        self.cos_scale_p = self.args.cosine_scale_p
        self.cos_scale_so = self.args.cosine_scale_so
        self.pairs = dset.pairs
        self.num_attrs, self.num_objs, self.num_pairs = len(self.dset.attrs), len(self.dset.objs), len(dset.pairs)
        self.num_attr_range = torch.arange(self.num_attrs)
        self.num_obj_range = torch.arange(self.num_objs)
        self.train_pair_range = torch.arange(len(dset.train_pairs))
        self.word_dim = get_word_dim(self.args)
        self.train_forward = self.train_forward_normal
        self.val_forward = self.val_forward_dotpr
        self.train_pair_acc = torch.zeros(len(dset.train_pairs))
        self.unseen_pairs = []

        for idx, i in enumerate(self.dset.pairs):
            if i not in self.dset.train_pairs:
                self.unseen_pairs.append(i)

        self.unseen_attrs, self.unseen_objs, _ = get_all_ids(self.unseen_pairs, dset.attr2idx, dset.obj2idx)
        self.val_attrs, self.val_objs, self.val_pairs = get_all_ids(self.dset.pairs, dset.attr2idx, dset.obj2idx)

        if args.train_only:
            self.train_attrs, self.train_objs, self.train_pairs = get_all_ids(self.dset.train_pairs, dset.attr2idx, dset.obj2idx)
        else:
            self.train_attrs, self.train_objs, self.train_pairs = self.val_attrs, self.val_objs, self.val_pairs

        self.args.fc_emb = self.args.fc_emb.split(',')
        layers = []
        for a in self.args.fc_emb:
            a = int(a)
            layers.append(a)

# ------------------------ Visual Space ----------------------------- #
        # base_prediction branch
        if args.nlayers:
            self.vp_embedder = MLP(dset.feat_dim, args.emb_dim, num_layers=args.nlayers,
                                      dropout=self.args.dropout,
                                      norm=self.args.norm, layers=layers, relu=True)

        # Disentangling architecture: attr&obj
        self.attr_dtg = MLP(inp_dim=dset.feat_dim, out_dim=dset.feat_dim, num_layers=2,
                                   dropout=True,
                                   norm=self.args.norm, layers=[2048], relu=True)
        self.v_attr_emb  = nn.Sequential(
            nn.Linear(dset.feat_dim, args.emb_dim),
            nn.ReLU()
        )
        self.obj_dtg = MLP(inp_dim=dset.feat_dim, out_dim=dset.feat_dim, num_layers=2,
                                  dropout=True,
                                  norm=self.args.norm, layers=[2048], relu=True)
        self.v_obj_emb = nn.Sequential(
            nn.Linear(dset.feat_dim, args.emb_dim),
            nn.ReLU()
        )
        # composing attr&obj->pair
        self.v_compose_emb = nn.Sequential(
            nn.Linear(dset.feat_dim * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(2048, args.emb_dim),
        )

# -------------- Textual Space ---------------------------- #
        self.w_attr_emb = nn.Sequential(
            nn.Linear(self.word_dim, args.emb_dim),
            nn.ReLU()
        )
        self.w_obj_emb = nn.Sequential(
            nn.Linear(self.word_dim, args.emb_dim),
            nn.ReLU()
        )
        self.w_p_emb = nn.Sequential(
            nn.Linear(self.word_dim * 2, args.emb_dim),
            nn.ReLU()
        )

        # init with word embeddings
        self.w_attr = nn.Embedding(len(dset.attrs), self.word_dim).to(device)
        self.w_obj = nn.Embedding(len(dset.objs), self.word_dim).to(device)
        if args.load_save_embeddings:
            attr_weights = ospj("./utils/", args.dataset + "_" + args.emb_init + '_attr-weights.t7')
            obj_weights = ospj("./utils/", args.dataset + "_" + args.emb_init + '_obj-weights.t7')
            if not os.path.exists(attr_weights or obj_weights):
                print("Generating embeddings...")
                attr_w = load_word_embeddings(args.emb_init, dset.attrs)
                obj_w = load_word_embeddings(args.emb_init, dset.objs)
                torch.save(attr_w, attr_weights)
                torch.save(obj_w, obj_weights)
            else:
                attr_w = torch.load(attr_weights)
                obj_w = torch.load(obj_weights)
            self.w_attr.weight.data.copy_(attr_w)
            self.w_obj.weight.data.copy_(obj_w)

            # similarity_map -> OS-OSP
            self.sim_obj_score = F.normalize(obj_w, dim=-1) @ F.normalize(obj_w.T, dim=-1)
            self.generate_sample_prob = nn.Linear(obj_w.size(0), obj_w.size(0))

        # collect
        obj2pair_mask = []
        for _obj in dset.objs:
            mask = [1 if _obj == obj else 0 for attr, obj in dset.pairs]
            obj2pair_mask.append(torch.BoolTensor(mask))
        self.obj2pair_mask = torch.stack(obj2pair_mask, 0)
        attr2pair_mask = []
        for _attr in dset.attrs:
            mask = [1 if _attr == attr else 0 for attr, obj in dset.pairs]
            attr2pair_mask.append(torch.BoolTensor(mask))
        self.attr2pair_mask = torch.stack(attr2pair_mask, 0)

    def compose(self, attrs, objs, use_cge=False):
        attrs, objs = self.w_attr(attrs), self.w_obj(objs)
        inputs = torch.cat([attrs, objs], 1)
        output = self.w_p_emb(inputs)
        output = F.normalize(output, dim=1)
        return output

    def attr_attack(self, input_img, attr_truth):
        # eval
        self.attr_dtg.eval()
        self.v_attr_emb.eval()
        self.w_attr.eval()
        self.w_attr_emb.eval()

        # forward
        attr_feats = self.attr_dtg(input_img)
        attr_feats = torch.nn.Parameter(attr_feats)
        attr_feats.requires_grad_()
        attr_visual_emb = self.v_attr_emb(attr_feats)

        fc_embedding_attr = self.w_attr_emb(self.w_attr(self.num_attr_range.to(device)))
        attr_pred = torch.matmul(F.normalize(attr_visual_emb, dim=1), F.normalize(fc_embedding_attr, dim=1).T)
        loss_attr = F.cross_entropy(self.cos_scale_so * attr_pred, attr_truth)

        # backward loss
        self.attr_dtg.zero_grad()
        self.v_attr_emb.zero_grad()
        self.w_attr.zero_grad()
        self.w_attr_emb.zero_grad()

        loss_attr.backward()

        # collect datagrad
        grad_attr_feats = attr_feats.grad.detach()
        # fgsm style attack
        index = torch.randint(0, len(epsilon_list), (1,))[0]
        epsilon = epsilon_list[index]
        adv_attr_feats = fgsm_attack(attr_feats, epsilon, grad_attr_feats)

        # eval
        self.attr_dtg.train()
        self.v_attr_emb.train()
        self.w_attr.train()
        self.w_attr_emb.train()

        return adv_attr_feats

    def obj_attack(self, input_img, obj_truth):
        # eval
        self.obj_dtg.eval()
        self.v_obj_emb.eval()
        self.w_obj.eval()
        self.w_obj_emb.eval()

        # forward
        obj_feats = self.obj_dtg(input_img)
        obj_feats = torch.nn.Parameter(obj_feats)
        obj_feats.requires_grad_()
        obj_visual_emb = self.v_obj_emb(obj_feats)

        fc_embedding_obj = self.w_obj_emb(self.w_obj(self.num_obj_range.to(device)))
        obj_pred = torch.matmul(F.normalize(obj_visual_emb, dim=1), F.normalize(fc_embedding_obj, dim=1).T)
        loss_obj = F.cross_entropy(self.cos_scale_so * obj_pred, obj_truth)

        # backward loss
        self.obj_dtg.zero_grad()
        self.v_obj_emb.zero_grad()
        self.w_obj.zero_grad()
        self.w_obj_emb.zero_grad()
        loss_obj.backward()

        # collect datagrad
        grad_obj_feats = obj_feats.grad.detach()
        # fgsm style attack
        index = torch.randint(0, len(epsilon_list), (1,))[0]
        epsilon = epsilon_list[index]
        adv_obj_feats = fgsm_attack(obj_feats, epsilon, grad_obj_feats)

        # eval
        self.obj_dtg.train()
        self.v_obj_emb.train()
        self.w_obj.train()
        self.w_obj_emb.train()

        return adv_obj_feats

    def __Label_smooth(self, inputs, targets, smoothing=0.1):
        assert 0 <= smoothing < 1
        label_all = []
        for idx in targets:
            _attr, _obj = self.dset.train_pairs[idx]  # 找到真实标签包含的attr和obj
            label = torch.LongTensor([1 if (_attr == attr or _obj == obj) else 0 for attr, obj in
                                      self.dset.train_pairs])  # 找到semi-positive
            label = (smoothing / (label.sum() - 1)) * label
            label[idx] = 1 - smoothing
            label_all.append(label)
        label_all = torch.stack(label_all, 0).to(device)
        log_logits = F.log_softmax(inputs, dim=1)
        temp = -1 * (log_logits * label_all)
        loss = (temp.sum(1).sum(0)) / temp.shape[0]
        return loss

    def __synthesize_compos(self, batch_size, strategy='random'):
        # compute sample probability
        sample_p_lb = []
        sample_up_lb = []
        for _ in range(batch_size):
            sample_p = np.random.choice(self.train_pair_range, p=F.softmax(1-self.train_pair_acc, dim=0).numpy())
            sample_up = np.random.choice(torch.arange(len(self.unseen_pairs)))
            sample_p_lb.append(sample_p)
            sample_up_lb.append(sample_up)
        p_attrs, p_objs = zip(*[self.dset.train_pairs[idx] for idx in sample_p_lb])
        up_attrs, up_objs = zip(*[self.unseen_pairs[idx] for idx in sample_up_lb])
        # get img feats
        if strategy == 'random':
            # seen
            obj_imgs = torch.zeros((batch_size, self.dset.feat_dim)).to(device)
            attr_imgs = torch.zeros((batch_size, self.dset.feat_dim)).to(device)
            for idx, obj in enumerate(p_objs):
                img_name = np.random.choice(self.dset.train_obj_affordance[obj])
                img_obj = self.dset.activations[img_name]
                obj_imgs[idx] = img_obj
            for idx, attr in enumerate(p_attrs):
                img_name = np.random.choice(self.dset.train_attr_affordance[attr])
                img_attr = self.dset.activations[img_name]
                attr_imgs[idx] = img_attr
            # compose
            fc_img_attr = self.attr_dtg(attr_imgs)
            fc_img_obj = self.obj_dtg(obj_imgs)
            X = F.normalize(torch.cat((fc_img_attr, fc_img_obj), dim=-1), dim=-1)  # Fusion by concat
            img_p_feats = F.normalize(self.v_compose_emb(X), dim=-1)

            # unseen
            obj_un_imgs = torch.zeros((batch_size, self.dset.feat_dim)).to(device)
            attr_un_imgs = torch.zeros((batch_size, self.dset.feat_dim)).to(device)
            for idx, obj in enumerate(up_objs):
                img_name = np.random.choice(self.dset.train_obj_affordance[obj])
                img_obj = self.dset.activations[img_name]
                obj_un_imgs[idx] = img_obj
            for idx, attr in enumerate(up_attrs):
                img_name = np.random.choice(self.dset.train_attr_affordance[attr])
                img_attr = self.dset.activations[img_name]
                attr_un_imgs[idx] = img_attr
            # compose unseen
            fc_img_attr = self.attr_dtg(attr_un_imgs)
            fc_img_obj = self.obj_dtg(obj_un_imgs)
            X = F.normalize(torch.cat((fc_img_attr, fc_img_obj), dim=-1), dim=-1)
            img_up_feats = F.normalize(self.v_compose_emb(X), dim=-1)

            return [img_p_feats, torch.tensor(sample_p_lb).to(device), img_up_feats,
                    torch.tensor(sample_up_lb).to(device)]

        elif strategy == 'similarity':
            obj_imgs = torch.zeros((batch_size, self.dset.feat_dim)).to(device)
            attr_imgs = torch.zeros((batch_size, self.dset.feat_dim)).to(device)
            # find img with obj
            for idx, obj in enumerate(p_objs):
                img_name = np.random.choice(self.dset.train_obj_affordance[obj])
                img_obj = self.dset.activations[img_name]
                obj_imgs[idx] = img_obj
            #  similarity based attr select
            for idx, _obj in enumerate(p_objs):
                # choose the _obj similar to obj under probability
                prob = F.softmax(self.sim_obj_score.to(device)[self.dset.obj2idx[_obj]], dim=0).to('cpu').numpy()
                sampled_obj_sim = np.random.choice(self.num_obj_range, p=prob)
                _attr = p_attrs[idx]
                while (1):
                    # if train_set contain the img with _attr under the _obj
                    if self.dset.sample_obj_affordance[self.dset.objs[sampled_obj_sim]][_attr]:
                        sampled_attr = np.random.choice(
                            self.dset.sample_obj_affordance[self.dset.objs[sampled_obj_sim]][_attr])
                        img_attr = self.dset.activations[sampled_attr]
                        attr_imgs[idx] = img_attr
                        break
                    else:
                        sampled_obj_sim = np.random.choice(self.num_obj_range, p=prob)
                        continue
            # compose
            fc_img_attr = self.attr_dtg(attr_imgs)
            fc_img_obj = self.obj_dtg(obj_imgs)
            X = F.normalize(torch.cat((fc_img_attr, fc_img_obj), dim=-1), dim=-1)
            img_p_feats = F.normalize(self.v_compose_emb(X), dim=-1)

            # unseen
            obj_un_imgs = torch.zeros((batch_size, self.dset.feat_dim)).to(device)
            attr_un_imgs = torch.zeros((batch_size, self.dset.feat_dim)).to(device)
            for idx, obj in enumerate(up_objs):
                img_name = np.random.choice(self.dset.train_obj_affordance[obj])
                img_obj = self.dset.activations[img_name]
                obj_un_imgs[idx] = img_obj
            for idx, _obj in enumerate(up_objs):
                # choose the _obj similar to obj under probability
                prob = F.softmax(self.sim_obj_score.to(device)[self.dset.obj2idx[_obj]], dim=0).to('cpu').numpy()
                sampled_obj_sim = np.random.choice(self.num_obj_range, p=prob)
                _attr = up_attrs[idx]
                while (1):
                    # if train_set contain the img with _attr under the _obj
                    if self.dset.sample_obj_affordance[self.dset.objs[sampled_obj_sim]][_attr]:
                        sampled_attr = np.random.choice(
                            self.dset.sample_obj_affordance[self.dset.objs[sampled_obj_sim]][_attr])
                        img_attr = self.dset.activations[sampled_attr]
                        attr_un_imgs[idx] = img_attr
                        break
                    else:
                        sampled_obj_sim = np.random.choice(self.num_obj_range, p=prob)
                        continue
            # compose unseen
            fc_img_attr = self.attr_dtg(attr_un_imgs)
            fc_img_obj = self.obj_dtg(obj_un_imgs)
            X = F.normalize(torch.cat((fc_img_attr, fc_img_obj), dim=-1), dim=-1)
            img_up_feats = F.normalize(self.v_compose_emb(X), dim=-1)

            return [img_p_feats, torch.tensor(sample_p_lb).to(device), img_up_feats,
                    torch.tensor(sample_up_lb).to(device)]
        else:
            raise NotImplementedError

    def train_forward_normal(self, x, epoch):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]
        B, C = img.size()
        # pair dict for score
        pair_sum = {f'class_{i}': [] for i in self.train_pairs}
        pair_right = {f'class_{i}': [] for i in self.train_pairs}
        # --pair
        if self.args.nlayers:
            img_feats = F.normalize(self.vp_embedder(img.detach()), dim=1)  # self.image_embedder(img)
        else:
            img_feats = (img.detach())
        pair_embed = self.compose(self.train_attrs, self.train_objs, use_cge=False)  # normalize in compose
        pair_pred = torch.matmul(img_feats, pair_embed.T)
        p_loss = self.__Label_smooth(self.cos_scale_p * pair_pred, pairs, smoothing=0.1) # F.cross_entropy(pair_pred, pairs)
        loss_all = p_loss

        # attr
        fc_img_attr = self.attr_dtg(img)
        fc_embedding_attr = self.w_attr_emb(self.w_attr(self.num_attr_range.to(device)))
        attr_pred = torch.matmul(F.normalize(self.v_attr_emb(fc_img_attr), dim=1), F.normalize(fc_embedding_attr, dim=1).T)
        loss_attr = F.cross_entropy(self.cos_scale_so * attr_pred, attrs)
        loss_all += self.args.attr_loss_w * loss_attr

        # --attack attr feats
        attr_attack = self.attr_attack(img, attrs)
        fc_embedding_attr = self.w_attr_emb(self.w_attr(self.num_attr_range.to(device)))
        attr_attack_pred = torch.matmul(F.normalize(self.v_attr_emb(attr_attack), dim=1), F.normalize(fc_embedding_attr, dim=1).T)
        loss_attr_attack = F.cross_entropy(self.cos_scale_so * attr_attack_pred, attrs)
        loss_all += self.args.attack_weight * loss_attr_attack

        # --obj--
        fc_img_obj = self.obj_dtg(img)
        fc_embedding_obj = self.w_obj_emb(self.w_obj(self.num_obj_range.to(device)))
        obj_pred = torch.matmul(F.normalize(self.v_obj_emb(fc_img_obj), dim=1), F.normalize(fc_embedding_obj, dim=1).T)
        loss_obj = F.cross_entropy(self.cos_scale_so * obj_pred, objs)
        loss_all += self.args.obj_loss_w * loss_obj

        # obj_attack
        obj_attack = self.obj_attack(img, objs)
        fc_embedding_obj = self.w_obj_emb(self.w_obj(self.num_obj_range.to(device)))
        obj_attack_pred = torch.matmul(F.normalize(self.v_obj_emb(obj_attack), dim=1), F.normalize(fc_embedding_obj, dim=1).T)
        loss_obj_attack = F.cross_entropy(self.cos_scale_so * obj_attack_pred, objs)
        loss_all += self.args.attack_weight * loss_obj_attack

        # compose
        X = F.normalize(torch.cat((fc_img_attr, fc_img_obj), dim=-1), dim=-1)
        img_feats = F.normalize(self.v_compose_emb(X), dim=-1)
        compose_pred = torch.matmul(img_feats, pair_embed.T)
        compose_loss = self.__Label_smooth(self.cos_scale_p * compose_pred, pairs, smoothing=0.1)
        loss_all += compose_loss

        # adv train
        X_adv = F.normalize(torch.cat((attr_attack, obj_attack), dim=-1), dim=-1)
        img_feats = F.normalize(self.v_compose_emb(X_adv), dim=-1)
        comp_adv_pred = torch.matmul(img_feats, pair_embed.T)
        comp_adv_loss = self.__Label_smooth(self.cos_scale_p * comp_adv_pred, pairs, smoothing=0.1)
        loss_all += self.args.attack_weight * comp_adv_loss

        if compose_pred.equal(comp_adv_pred):
            cp_kl = 0
        else:
            cp_kl = consistency_loss(compose_pred, comp_adv_pred, 'KL3')

        loss_all += self.args.attack_weight * cp_kl

        # synthesize hard compositions
        if self.args.use_os_osp and epoch >= self.args.start_syn and not self.args.open_world:
            with torch.no_grad():
                syn_dt = self.__synthesize_compos(batch_size=B, strategy='similarity')
                syn_p_feats, syn_p_lb, syn_up_feats, syn_up_lb = syn_dt[0], syn_dt[1], syn_dt[2], syn_dt[3]
            # compute_loss
            syn_p_pred = torch.matmul(syn_p_feats, pair_embed.T)
            syn_p_loss = self.__Label_smooth(self.cos_scale_p * syn_p_pred, syn_p_lb, smoothing=0.1)
            loss_all += syn_p_loss
            unseen_pair_embed = self.compose(self.unseen_attrs, self.unseen_objs)
            syn_up_pred = torch.matmul(syn_up_feats, unseen_pair_embed.T)
            syn_up_loss = F.cross_entropy(self.cos_scale_p * syn_up_pred, syn_up_lb)
            loss_all += syn_up_loss

        # collect pair predict
        pair_top1 = torch.max(pair_pred, dim=1)[1]
        for index, p_t in enumerate(pairs):
            pair_sum[f'class_{p_t}'].append(p_t)
            if pair_top1[index] == p_t:
                pair_right[f'class_{p_t}'].append(p_t)
        # --use_calibration
        if self.args.use_calibration:
            with torch.no_grad():
                unseen_pair_embed = self.compose(self.unseen_attrs, self.unseen_objs).permute(1, 0)
                unseen_index = found_affinity_unseen_paris(pair_embed.detach(), unseen_pair_embed.detach().T)
                unseen_index = unseen_index[pairs].detach()
            unseen_pair_pred = torch.matmul(img_feats, unseen_pair_embed)
            loss_unseen = F.cross_entropy(self.cos_scale_p * unseen_pair_pred, unseen_index).mean()
            loss_all += self.args.calibration_weights * loss_unseen

            return loss_all, [pair_sum, pair_right], None, None

    def val_forward_dotpr(self, x):
        img = x[0]  # x[0]是resnet存储或者提取的图片特征，x[1-3]分别是属性、物体和组合的标签。
        # --pair--
        if self.args.nlayers:
            img_feats = F.normalize(self.vp_embedder(img), dim=1)
        else:
            img_feats = (img)
        pair_embeds = self.compose(self.val_attrs, self.val_objs)
        score = torch.matmul(img_feats, pair_embeds.T)

        # attr
        attr_vs = self.attr_dtg(img)
        attr_vs_emb = F.normalize(self.v_attr_emb(attr_vs), dim=1)
        fc_embedding_attr = F.normalize(self.w_attr_emb(self.w_attr(self.num_attr_range.to(device))),
                                        dim=1)
        attr_score = torch.matmul(attr_vs_emb, fc_embedding_attr.T)

        # --obj--
        obj_vs = self.obj_dtg(img)
        obj_vs_emb = F.normalize(self.v_obj_emb(obj_vs), dim=1)
        fc_embedding_obj = F.normalize(self.w_obj_emb(self.w_obj(self.num_obj_range.to(device))), dim=1)
        obj_score = torch.matmul(obj_vs_emb, fc_embedding_obj.T)

        # compose
        X = F.normalize(torch.cat((attr_vs, obj_vs), dim=-1), dim=-1)
        img_feats = F.normalize(self.v_compose_emb(X), dim=-1)
        compose_score = torch.matmul(img_feats, pair_embeds.T)

        # Add scores to pair_score
        attr_score2pair = torch.matmul(attr_score, self.attr2pair_mask.float().to(device)).to(device)
        obj_score2pair = torch.matmul(obj_score, self.obj2pair_mask.float().to(device)).to(device)
        score += attr_score2pair + obj_score2pair + compose_score

        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:, self.dset.all_pair2idx[pair]]  # 取出对应每个组合，batch_size个样本的得分

        return None, None, None, scores, attr_score, obj_score

    def forward(self, x, epoch):
        if self.training:
            loss, pred, pred_attr, pred_obj = self.train_forward(x, epoch)
            return loss, pred, pred_attr, pred_obj
        else:
            with torch.no_grad():
                loss, loss_attr, loss_obj, pred, pred_attr, pred_obj = self.val_forward(x)
                return loss, loss_attr, loss_obj, pred, pred_attr, pred_obj
