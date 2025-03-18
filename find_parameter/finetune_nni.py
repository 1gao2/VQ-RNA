import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import loguru
import numpy as np
import torch
import torch.nn as nn
# import F
import torch.nn.functional as F
import yaml
# set cuda visible devices
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from echo_logger import print_debug, print_info, dumps_json
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from _config import *
from data_process.compound_tools import CompoundKit
from data_process.data_collator import collator_finetune_pkl
from data_process.function_group_constant import nfg
from data_process.split import create_splitter
from datasets.dataloader_pkl import FinetuneDataset as FinetuneDataset_pkl
from models.loong import Loong
from utils.global_var_util import *
from utils.loss_util import bce_loss, get_balanced_atom_fg_loss, use_balanced_atom_fg_loss
from utils.metric_util import compute_reg_metric, compute_cls_metric_tensor
from utils.public_util import set_seed, EarlyStopping
from utils.scheduler_util import *
from utils.userconfig_util import config_current_user, config_dataset_form, get_dataset_form, drop_last_flag

# warnings.filterwarnings("ignore")

np.set_printoptions(threshold=10)
torch.set_printoptions(threshold=10)
# torch.set_float32_matmul_precision('high')
# set default device cuda
# torch.set_default_device('cuda')

import nni

torch.set_printoptions(sci_mode=False, precision=2, linewidth=400, threshold=1000000000)

# noinspection SpellCheckingInspection
class Trainer(object):
    def __init__(self, config, file_path):
        self.imbalance_ratio = None
        self.config = config

        self.train_loader, self.val_loader, self.test_loader = self.get_data_loaders()
        self.net = self._get_net()
        # print_info("Compiling model...")
        # self.net = torch.compile(self.net)
        # print_info("Model compiled!")
        self.criterion = self._get_loss_fn().cuda()
        self.optim = self._get_optim()
        self.lr_scheduler = self._get_lr_scheduler()
        loguru.logger.info(f"Optimizer: {self.optim}")
        if config['checkpoint'] and GlobalVar.use_ckpt:
            self.load_ckpt(self.config['checkpoint'])
        else:
            loguru.logger.warning("No checkpoint loaded!")
            self.start_epoch = 1
            self.optim_steps = 0
            self.best_metric = -np.inf if config['task'] == 'classification' else np.inf
            if not os.path.exists('../train_result'):
                os.makedirs('../train_result')
            self.writer = SummaryWriter('../train_result/{}/{}_{}_{}_{}_{}_{}'.format(
                'finetune_result', self.config['task_name'], self.config['seed'], self.config['split_type'],
                self.config['optim']['init_lr'],
                self.config['batch_size'], datetime.now().strftime('%b%d_%H:%M:%S')
            ))
        self.txtfile = os.path.join(self.writer.log_dir, 'record.txt')
        copyfile(file_path, self.writer.log_dir)

        self.batch_considered = 200

        self.loss_init = torch.zeros(3, self.batch_considered, device='cuda')
        self.loss_last = torch.zeros(3, self.batch_considered // 10, device='cuda')
        self.loss_last2 = torch.zeros(3, self.batch_considered // 10, device='cuda')
        self.cur_loss_step = torch.zeros(1, dtype=torch.long, device='cuda')
        # self.register_buffer('loss_init', loss_init)
        # self.register_buffer('loss_last', loss_last)
        # self.register_buffer('loss_last2', loss_last2)
        # self.register_buffer('cur_loss_step', cur_loss_step)

    def calc_mt_loss(self, loss_list):
        loss_list = torch.stack(loss_list)

        if self.cur_loss_step == 0:
            self.loss_init[:, 0] = loss_list.detach()
            self.loss_last2[:, 0] = loss_list.detach()
            self.cur_loss_step += 1
            loss_t = (loss_list / self.loss_init[:, 0]).mean()

        elif self.cur_loss_step == 1:

            self.loss_last[:, 0] = loss_list.detach()
            self.loss_init[:, 1] = loss_list.detach()
            self.cur_loss_step += 1
            loss_t = (loss_list / self.loss_init[:, :2].mean(dim=-1)).mean()

        else:
            cur_loss_init = self.loss_init[:, :self.cur_loss_step].mean(dim=-1)
            cur_loss_last = self.loss_last[:, :self.cur_loss_step - 1].mean(dim=-1)
            cur_loss_last2 = self.loss_last2[:, :self.cur_loss_step - 1].mean(dim=-1)
            w = F.softmax(cur_loss_last / cur_loss_last2, dim=-1).detach()
            loss_t = (loss_list / cur_loss_init * w).sum()

            cur_init_idx = self.cur_loss_step.item() % self.batch_considered
            self.loss_init[:, cur_init_idx] = loss_list.detach()

            cur_loss_last2_step = (self.cur_loss_step.item() - 1) % (self.batch_considered // 10)
            self.loss_last2[:, cur_loss_last2_step] = self.loss_last[:, cur_loss_last2_step - 1]
            self.loss_last[:, cur_loss_last2_step] = loss_list.detach()
            self.cur_loss_step += 1
        return loss_t

    def get_data_loaders(self):
        dataset = FinetuneDataset_pkl(root=self.config['root'], task_name=self.config['task_name'])
        splitter = create_splitter(self.config['split_type'], self.config['seed'])
        train_dataset, val_dataset, test_dataset = splitter.split(dataset, self.config['task_name'])

        # self.imbalance_ratio = ((dataset.data.label == -1).sum()) / ((dataset.data.label == 1).sum())
        num_workers = self.config['dataloader_num_workers']
        bsz = self.config['batch_size']
        train_loader = DataLoader(train_dataset,
                                  batch_size=bsz,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  collate_fn=collator_finetune_pkl,
                                  pin_memory=True,
                                  drop_last=drop_last_flag(len(train_dataset), bsz))
        val_loader = DataLoader(val_dataset,
                                batch_size=bsz,
                                shuffle=False,
                                num_workers=num_workers,
                                collate_fn=collator_finetune_pkl,
                                pin_memory=True,
                                drop_last=drop_last_flag(len(val_dataset), bsz))
        test_loader = DataLoader(test_dataset,
                                 batch_size=bsz,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 collate_fn=collator_finetune_pkl,
                                 drop_last=drop_last_flag(len(test_dataset), bsz))

        return train_loader, val_loader, test_loader

    def _get_net(self):
        model = Loong(mode=config['mode'], atom_names=CompoundKit.atom_vocab_dict.keys(),
                      bond_names=CompoundKit.bond_vocab_dict.keys(), atom_embed_dim=config['model']['atom_embed_dim'],
                      bond_embed_dim=config['model']['bond_embed_dim'],
                      angle_embed_dim=config['model']['angle_embed_dim'],
                      num_kernel=config['model']['num_kernel'], layer_num=config['model']['layer_num'],
                      num_heads=config['model']['num_heads'],
                      atom_FG_class=nfg() + 1, bond_FG_class=config['model']['bond_FG_class'],
                      hidden_size=config['model']['hidden_size'], num_tasks=config['num_tasks'],
                      cross_layers=config['cross_layers']).cuda()
        model_name = 'model.pth'
        if self.config['pretrain_model_path'] != 'None':
            model_path = os.path.join(self.config['pretrain_model_path'], model_name)
            state_dict = torch.load(model_path, map_location='cuda')
            model.model.load_model_state_dict(state_dict['model'])
            print("Loading pretrain model from", os.path.join(self.config['pretrain_model_path'], model_name))
        if GlobalVar.parallel_train:
            model = nn.DataParallel(model)
        return model

    def _get_loss_fn(self):
        loss_type = self.config['loss_type']
        if loss_type == 'bce':
            return bce_loss()
        elif loss_type == 'wb_bce':
            ratio = self.imbalance_ratio
            return bce_loss(weights=[1.0, ratio])
        elif loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'l1':
            return nn.L1Loss()
        else:
            raise ValueError('not supported loss function!')

    def _get_optim(self):
        optim_type = self.config['optim']['type']
        lr = self.config['optim']['init_lr']
        weight_decay = self.config['optim']['weight_decay']
        freezing_layers_regex = []
        freezing_layers = []
        if GlobalVar.freeze_layers > 0 and GlobalVar.use_ckpt:
            for i in range(GlobalVar.freeze_layers):
                freezing_layers_regex.append(f'EncoderAtomList.{i}')
                freezing_layers_regex.append(f'EncoderBondList.{i}')
        for name, param in self.net.named_parameters():
            if GlobalVar.freeze_layers > 0:
                for part_ in freezing_layers_regex:
                    if part_ in name:
                        param.requires_grad = False
                        freezing_layers.append(name)
                        break

        base_params = list(
            map(lambda x: x[1], list(filter(lambda kv: kv[0] not in freezing_layers, self.net.named_parameters()))))
        base_params_names = list(
            map(lambda x: x[0], list(filter(lambda kv: kv[0] not in freezing_layers, self.net.named_parameters()))))

        model_params = [{'params': base_params}]
        for p_name in freezing_layers:
            assert p_name in [p[0] for p in self.net.named_parameters()]
            assert p_name not in base_params_names
        if optim_type == 'adam':
            return torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
        elif optim_type == 'rms':
            return torch.optim.RMSprop(model_params, lr=lr, weight_decay=weight_decay)
        elif optim_type == 'sgd':
            momentum = self.config['optim']['momentum'] if 'momentum' in self.config['optim'] else 0
            return torch.optim.SGD(model_params, lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError('not supported optimizer!')

    def _get_lr_scheduler(self):
        scheduler_type = self.config['lr_scheduler']['type']
        init_lr = self.config['lr_scheduler']['start_lr']
        warm_up_epoch = self.config['lr_scheduler']['warm_up_epoch']

        if scheduler_type == 'linear':
            return LinearSche(self.config['epochs'], warm_up_end_lr=self.config['optim']['init_lr'], init_lr=init_lr,
                              warm_up_epoch=warm_up_epoch)
        elif scheduler_type == 'square':
            return SquareSche(self.config['epochs'], warm_up_end_lr=self.config['optim']['init_lr'], init_lr=init_lr,
                              warm_up_epoch=warm_up_epoch)
        elif scheduler_type == 'cos':
            return CosSche(self.config['epochs'], warm_up_end_lr=self.config['optim']['init_lr'], init_lr=init_lr,
                           warm_up_epoch=warm_up_epoch)
        elif scheduler_type == 'None':
            return None
        else:
            raise ValueError('not supported learning rate scheduler!')

    def _step(self, model, batch: Dict[str, Tensor]):
        dataset_form: str = get_dataset_form()
        max_fn_group_edge_type_additional_importance_score = GlobalVar.max_fn_group_edge_type_additional_importance_score
        fg_edge_type_num = GlobalVar.fg_edge_type_num
        function_group_type_nmber = GlobalVar.fg_number
        pred_dict: Dict = model(batch)
        pred = pred_dict['graph_feature']
        finger = pred_dict['finger_feature']
        atom_fg = pred_dict['atom_fg']
        bond_fg = pred_dict['bond_fg']
        function_group_index = batch['function_group_index']
        function_group_bond_index = batch['function_group_bond_index']

        # mask_atom_fg = (function_group_index != -1)
        # atom_fg = atom_fg[mask_atom_fg]
        # function_group_index = function_group_index[mask_atom_fg]

        # mask_bond_fg = (function_group_bond_index != -1)
        # bond_fg = bond_fg[mask_bond_fg]
        # function_group_bond_index = function_group_bond_index[mask_bond_fg]
        #
        # loss_f_bond_fg = nn.CrossEntropyLoss()
        # loss_bond_fg = loss_f_bond_fg(bond_fg.view(-1, fg_edge_type_num),
        #                               (function_group_bond_index.long()).view(-1))
        loss_style = GlobalVar.loss_style

        if loss_style != LossStyle.loong_and_finger_loss and loss_style != LossStyle.loong_loss:
            if use_balanced_atom_fg_loss(loss_style):
                loss_atom_fg = get_balanced_atom_fg_loss(atom_fg, function_group_index,
                                                         loss_f_atom_fg=F.binary_cross_entropy_with_logits)
            else:
                loss_atom_fg = F.binary_cross_entropy_with_logits(atom_fg.view(-1, function_group_type_nmber),
                                                                  (function_group_index + 0.0).view(-1,
                                                                                                    function_group_type_nmber))
        loss_finger = F.binary_cross_entropy_with_logits(finger, batch['morgan2048_fp'].float())
        # pred size: [batch_size, 1]
        if self.config['task'] == 'classification':
            label: Tensor = batch['label']
            # print(label.shape)
            # label = label[:, 0::2]
            # print(label.shape)
            if dataset_form == 'pyg':
                is_valid: Tensor = label ** 2 > 0
                label = ((label + 1.0) / 2).view(pred.shape)
            elif dataset_form == 'pkl':
                is_valid: Tensor = (label >= 0)
                label = (label + 0.0).view(pred.shape)
            else:
                raise ValueError('not supported dataset form!')

            loss = self.criterion(pred, label)
            # print('pred=', pred.shape)
            loss = torch.where(is_valid, loss, torch.zeros(loss.shape, device='cuda').to(loss.dtype))
            loss = torch.sum(loss) / torch.sum(is_valid)
        else:
            loss = self.criterion(pred, batch['label'].float())
        if loss_style == LossStyle.loong_loss:
            return loss, pred
        elif loss_style == LossStyle.loong_and_finger_loss:
            return loss + loss_finger, pred
        elif loss_style == LossStyle.loong_and_balanced_fg_loss:
            return loss + loss_atom_fg + loss_bond_fg, pred
        elif loss_style == LossStyle.loong_and_yw_balanced_fg_loss:
            return self.calc_mt_loss([loss, loss_atom_fg, loss_bond_fg]), pred
        elif loss_style == LossStyle.loong_and_finger_and_yw_balanced_fg_loss or \
                loss_style == LossStyle.loong_and_finger_and_yw_unbalanced_fg_loss:
            return self.calc_mt_loss([loss, loss_finger, loss_atom_fg]), pred
        elif loss_style == LossStyle.loong_and_finger_and_balanced_fg_loss or \
                loss_style == LossStyle.loong_and_finger_and_unbalanced_fg_loss:
            return loss + loss_finger + loss_atom_fg + loss_bond_fg, pred
        else:
            raise ValueError('not supported loss style!')

    def _train_step(self):
        self.net.train()
        num_data = 0
        train_loss = 0
        # y_pred = []
        # y_true = []
        y_pred: Tensor = Tensor().to('cuda')
        y_true: Tensor = Tensor().to('cuda')
        for _, batch in tqdm(enumerate(self.train_loader)):
            self.optim.zero_grad()
            batch = {key: value.to('cuda') for key, value in batch.items()
                     if value is not None and not isinstance(value, list)}
            batch['edge_weight'] = None

            loss, pred = self._step(self.net, batch)
            train_loss += loss.item()
            self.writer.add_scalar('train_loss', loss, global_step=self.optim_steps)

            # y_pred.extend(pred)
            # y_true.extend(batch['label'])
            # we cannot extend tensor
            y_pred = torch.cat([y_pred, pred])
            y_true = torch.cat([y_true, batch['label']])
            loss.backward()
            self.optim.step()
            num_data += 1
            self.optim_steps += 1

        train_loss /= num_data
        torch.cuda.empty_cache()
        if self.config['task'] == 'regression':
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                mae, _ = compute_reg_metric(y_true, y_pred)
                return train_loss, mae
            else:
                _, rmse = compute_reg_metric(y_true, y_pred)
                return train_loss, rmse
        elif self.config['task'] == 'classification':
            roc_auc = compute_cls_metric_tensor(y_true, y_pred)

            return train_loss, roc_auc

    def _valid_step(self, valid_loader):
        self.net.eval()
        y_pred = Tensor().to('cuda')
        y_true = Tensor().to('cuda')
        valid_loss = 0
        num_data = 0
        for batch in valid_loader:
            batch = {key: value.to('cuda') for key, value in batch.items()
                     if value is not None and not isinstance(value, list)}
            batch['edge_weight'] = None
            with torch.no_grad():
                loss, pred = self._step(self.net, batch)
            valid_loss += loss.item()
            num_data += 1
            # y_pred.extend(pred)
            # y_true.extend(batch['label'])
            # we cannot extend tensor
            y_pred = torch.cat([y_pred, pred])
            y_true = torch.cat([y_true, batch['label']])

        valid_loss /= num_data

        if self.config['task'] == 'regression':
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                mae, _ = compute_reg_metric(y_true, y_pred)
                return valid_loss, mae
            else:
                _, rmse = compute_reg_metric(y_true, y_pred)
                return valid_loss, rmse
        elif self.config['task'] == 'classification':
            roc_auc = compute_cls_metric_tensor(y_true, y_pred)
            return valid_loss, roc_auc

    def save_ckpt(self, epoch):
        checkpoint = {
            "net": self.net.state_dict(),
            'optimizer': self.optim.state_dict(),
            "epoch": epoch,
            'best_metric': self.best_metric,
            'optim_steps': self.optim_steps
        }
        path = os.path.join(self.writer.log_dir, 'checkpoint')
        os.makedirs(path, exist_ok=True)
        torch.save(checkpoint, os.path.join(self.writer.log_dir, 'checkpoint', 'model_{}.pth'.format(epoch)))

    def load_ckpt(self, load_pth):
        if GlobalVar.use_ckpt:
            print(f'load model from {load_pth}')
            checkpoint = torch.load(load_pth, map_location='cuda')['model']
            new_model_dict = self.net.state_dict()
            new_model_keys = set(list(new_model_dict.keys()))

            pretrained_dict = {'.'.join(k.split('.')): v for k, v in checkpoint.items()}
            pretrained_keys = set(list('.'.join(k.split('.')) for k in checkpoint.keys()))
            is_dp = model_is_dp(pretrained_keys)
            if is_dp:
                # rmv 'module.' prefix
                pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
                pretrained_keys = set(list(k[7:] for k in pretrained_keys))
                # only update the same keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_keys}
            diff_, same_ = new_model_keys - pretrained_keys, new_model_keys & pretrained_keys
            not_used_keys = pretrained_keys - new_model_keys
            # print_info("Not Loaded Keys: ", dumps_json(sorted(list(diff_))))
            print_info("Not Loaded Keys: (Keys in model but not in pretrained model)", dumps_json(sorted(list(diff_))))
            print_info("Same Keys (Loaded Keys), Though maybe not updated by optimizer: ", dumps_json(sorted(list(same_))))
            print_info("Not Used Keys (Keys in pretrained model but not in model): ", dumps_json(sorted(list(not_used_keys))))
            new_model_dict.update(pretrained_dict)
            self.net.load_state_dict(new_model_dict)
            if 'dist_bar' in checkpoint:
                GlobalVar.dist_bar = checkpoint['dist_bar'].cpu().numpy()

        self.start_epoch = 1
        self.optim_steps = 0
        self.best_metric = -np.inf if config['task'] == 'classification' else np.inf
        self.writer = SummaryWriter('{}/{}_{}_{}_{}_{}_{}'.format(
            'finetune_result', self.config['task_name'], self.config['seed'], self.config['split_type'],
            self.config['optim']['init_lr'],
            self.config['batch_size'], datetime.now().strftime('%b%d_%H:%M')))

    def train(self):
        # print(self.config)
        # print(self.config['DownstreamModel']['dropout'])
        write_record(self.txtfile, self.config)
        self.net = self.net.to('cuda')
        # 设置模型并行
        # if self.config['DP']:
        #     self.net = torch.nn.DataParallel(self.net)
        # 设置早停
        mode = 'lower' if self.config['task'] == 'regression' else 'higher'
        stopper = EarlyStopping(mode=mode, patience=GlobalVar.patience,
                                filename=os.path.join(self.writer.log_dir, 'model.pth'))

        val_metric_list, test_metric_list = [], []
        for i in range(self.start_epoch, self.config['epochs'] + 1):
            print(GlobalVar.dist_bar)
        # for i in range(self.start_epoch, 2):
            if self.config['lr_scheduler']['type'] in ['cos', 'square', 'linear']:
                self.lr_scheduler.adjust_lr(self.optim, i)
            # print("Epoch {} cur_lr {}".format(i, self.optim.param_groups[1]['lr']))
            train_loss, train_metric = self._train_step()
            valid_loss, valid_metric = self._valid_step(self.val_loader)
            test_loss, test_metric = self._valid_step(self.test_loader)
            val_metric_list.append(valid_metric)
            test_metric_list.append(test_metric)
            if stopper.step(valid_metric, self.net, test_score=test_metric):
                stopper.report_final_results(i_epoch=i)
                break
            if self.config['save_model'] == 'best_valid':
                if (self.config['task'] == 'regression' and (self.best_metric > valid_metric)) or (
                        self.config['task'] == 'classification' and (self.best_metric < valid_metric)):
                    self.best_metric = valid_metric
                    if self.config['DP']:
                        if RoutineControl.save_best_ckpt:
                            torch.save(self.net.module.state_dict(), os.path.join(self.writer.log_dir, 'model.pth'))
                    else:
                        model_dict = {'model': self.net.state_dict()}
                        if RoutineControl.save_best_ckpt:
                            torch.save(model_dict, os.path.join(self.writer.log_dir, 'model.pth'))
            elif self.config['save_model'] == 'each':
                if self.config['DP']:
                    if RoutineControl.save_best_ckpt:
                        torch.save(self.net.module.state_dict(),
                                   os.path.join(self.writer.log_dir, 'model_{}.pth'.format(i)))
                else:
                    if RoutineControl.save_best_ckpt:
                        torch.save(self.net.state_dict(), os.path.join(self.writer.log_dir, 'model_{}.pth'.format(i)))
            self.writer.add_scalar('valid_loss', valid_loss, global_step=i)
            self.writer.add_scalar('test_loss', test_loss, global_step=i)

            if config['task'] == 'classification':
                print(f'train_loss:{train_loss} valid_loss:{valid_loss} test_loss:{test_loss}\n'
                      f'train_auc:{train_metric} valid_auc:{valid_metric} test_auc:{test_metric}')
            else:
                if self.config["task_name"] in ['qm7', 'qm8', 'qm9']:
                    print(f'train_loss:{train_loss} valid_loss:{valid_loss} test_loss:{test_loss}\n'
                          f'train_mae:{train_metric} valid_mae:{valid_metric} test_mae:{test_metric}')
                else:
                    print(f'train_loss:{train_loss} valid_loss:{valid_loss} test_loss:{test_loss}\n'
                          f'train_rmse:{train_metric} valid_rmse:{valid_metric} test_rmse:{test_metric}')
            write_record(self.txtfile,
                         f'epoch:{i}\n'
                         f'train_loss:{train_loss} valid_loss:{valid_loss} test_loss:{test_loss}\n'
                         f'train_metric:{train_metric} valid_metric:{valid_metric} test_metric:{test_metric}')
            if i % self.config['save_ckpt'] == 0 and RoutineControl.save_best_ckpt:
                self.save_ckpt(i)
            nni.report_intermediate_result(test_metric)

        if config['task'] == 'classification':
            best_val_metric = np.max(val_metric_list)
            best_test_metric = np.max(test_metric_list)
            true_test_metric = test_metric_list[np.argmax(val_metric_list)]

        elif config['task'] == 'regression':
            best_val_metric = np.min(val_metric_list)
            best_test_metric = np.min(test_metric_list)
            true_test_metric = test_metric_list[np.argmin(val_metric_list)]
        else:
            raise ValueError('only supported classification or regression!')

        print(f'best_val_metric:{best_val_metric}\t'
              f'best_test_metric:{best_test_metric}\t'
              f'true_test_metric:{true_test_metric}')

        write_record(self.txtfile, f'best_val_metric:{best_val_metric}\t'
                                   f'best_test_metric:{best_test_metric}\t'
                                   f'true_test_metric:{true_test_metric}')
        nni.report_final_result(true_test_metric)


if __name__ == '__main__':
    params_nni = {
        'seed': 2,
        'batch_size': 32,
        'task': 'bace',
        'init_lr': 0.00005,
        'weight_decay': 0.0001,
        'dataloader_num_workers': 4,
        'hyper_node_init_with_morgan': False,
        "loss_style": LossStyle.loong_and_finger_and_yw_balanced_fg_loss,
        "embedding_style": "more",
        "transformer_dim": 512,
        "ffn_dim": 256,
        "num_heads": 16,
        'cross_layers': 1000,
        "patience": 10,
        'use_ckpt': True,
        # 'checkpoint': '/root/autodl-tmp/code/Loong-pretrain/pretrain_result/pretrain_plus_bin/bbbp_8_Apr19_00_19/model.pth',
        # 'checkpoint': '/root/autodl-tmp/data/fg_finger_sp_yw.pth',
        # 'ckpt_': 'fg_fg_number_finger_mask__all',
        'dropout': 0.1,
        'ckpt_': 'fg_finger_sp_angle_yw',
        'freeze_layers': 0,
        'layer_num': 6,
        'optim_type': 'adam/weight_decay=0.0001',
        'dist_bar_type': '3d',
        'dist_bar_start': 30,
        'dist_bar_end': 40,
    }
    # GlobalVar.is_mol_net_tasks = True
    tuner_params = nni.get_next_parameter()  # 获得下一组搜索空间中的参数
    params_nni.update(tuner_params)  # 更新参数
    print(params_nni)
    # if contains key ckpt_, then checkpoint = '/data/wangding/projects/Loong/model_ckp/{ckpt_}.pth'
    if 'ckpt_' in params_nni:
        # params_nni['checkpoint'] = f'/data/wangding/projects/train_result/pretrain_result/final_era/{params_nni["ckpt_"]}.pth'
        params_nni['checkpoint'] = f'/root/autodl-tmp/data/{params_nni["ckpt_"]}.pth'
        # params_nni['checkpoint'] = f'/root/autodl-tmp/data/models/{params_nni["ckpt_"]}.pth'
        if params_nni['ckpt_'] == 'None':
            params_nni['use_ckpt'] = False
            print("params_nni['use_ckpt'] set to False, because params_nni['ckpt_'] == 'None'")
    if params_nni['task'] == 'sider':
        params_nni['batch_size'] = 10
    if params_nni['task'] in REGRESSION_TASK_NAMES or params_nni['task'] == 'clintox':
        params_nni["patience"] = 30
    path = Path(pdir) / "config" / "config_finetune.yaml"
    config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    config = config_current_user('autodl', config)  # TODO: CHANGE IT ACCORDING TO YOUR NEEDS
    config = config_dataset_form('pkl', config)
    config['task_name'] = params_nni['task']
    config['batch_size'] = params_nni['batch_size']
    config['seed'] = params_nni['seed']
    config['dropout'] = params_nni['dropout']
    DEFAULTS.DROP_RATE = params_nni['dropout']
    config['dataloader_num_workers'] = params_nni['dataloader_num_workers']
    config['optim']['init_lr'] = params_nni['init_lr']
    config['cross_layers'] = params_nni['cross_layers']  #
    config['checkpoint'] = params_nni['checkpoint']
    config['optim']['type'] = params_nni['optim_type'].split('/')[0] if '/' in params_nni['optim_type'] else params_nni[
        'optim_type']
    config['optim']['weight_decay'] = float(params_nni['optim_type'].split('=')[1]) if '/' in params_nni[
        'optim_type'] else params_nni['weight_decay']
    config['optim']['momentum'] = float(params_nni['optim_type'].split('=')[1]) if '/' in params_nni[
        'optim_type'] else 0
    set_seed(config['seed'])
    config = get_downstream_task_names(config)
    GlobalVar.patience = params_nni['patience']
    config['patience'] = GlobalVar.patience
    RoutineControl.save_best_ckpt = False
    GlobalVar.hyper_node_init_with_morgan = params_nni['hyper_node_init_with_morgan']
    GlobalVar.loss_style = params_nni['loss_style']
    # GlobalVar.dist_bar = [params_nni['dist_bar_start']]
    # set_dist_bar_three(params_nni['dist_bar_start'], params_nni['dist_bar_end'])
    set_dist_bar_two(params_nni['dist_bar_start'], params_nni['dist_bar_end'])
    # GlobalVar.dist_bar_type = params_nni['dist_bar_type']
    # set_dist_bar_all_100(params_nni['dist_bar_start'], params_nni['dist_bar_middle'], params_nni['dist_bar_end'])
    print(GlobalVar.dist_bar)
    if GlobalVar.dist_bar is None:
        sys.exit(0)
    GlobalVar.embedding_style = params_nni['embedding_style']
    GlobalVar.transformer_dim = params_nni['transformer_dim']
    GlobalVar.ffn_dim = params_nni['ffn_dim']
    GlobalVar.num_heads = params_nni['num_heads']
    GlobalVar.use_ckpt = params_nni['use_ckpt']
    GlobalVar.freeze_layers = params_nni['freeze_layers']
    if GlobalVar.freeze_layers < 0:
        GlobalVar.use_ckpt = False
        print("GlobalVar.use_ckpt set to False, because GlobalVar.freeze_layers < 0")
    GlobalVar.parallel_train = False

    config['model']['atom_embed_dim'] = GlobalVar.transformer_dim
    config['model']['bond_embed_dim'] = GlobalVar.transformer_dim
    config['model']['hidden_size'] = GlobalVar.ffn_dim
    config['model']['num_heads'] = GlobalVar.num_heads
    config['model']['layer_num'] = params_nni['layer_num']
    config['fg_num_'] = nfg() + 1
    config['freeze_layers'] = GlobalVar.freeze_layers
    config['loss_style'] = GlobalVar.loss_style
    print_debug(dumps_json(config))

    trainer = Trainer(config, path)
    trainer.train()
