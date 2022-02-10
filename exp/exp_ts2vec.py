from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.ts2vec.encoder import TSEncoder
from models.ts2vec.losses import hierarchical_contrastive_loss
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split

import os
import time

import warnings
warnings.filterwarnings('ignore')


def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]


def fit_ridge(train_features, train_y, valid_features, valid_y, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if train_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            train_features, train_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        train_features = split[0]
        train_y = split[2]
    if valid_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            valid_features, valid_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        valid_features = split[0]
        valid_y = split[2]
    alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    valid_results = []
    for alpha in alphas:
        lr = Ridge(alpha=alpha).fit(train_features, train_y)
        valid_pred = lr.predict(valid_features)
        score = np.sqrt(((valid_pred - valid_y) ** 2).mean()) + np.abs(valid_pred - valid_y).mean()
        valid_results.append(score)
    best_alpha = alphas[np.argmin(valid_results)]

    lr = Ridge(alpha=best_alpha)
    lr.fit(train_features, train_y)
    return lr


class Exp_TS2Vec(Exp_Basic):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self._net = TSEncoder(input_dims=self.args.enc_in + 7, output_dims=self.args.c_out,
                              hidden_dims=64, depth=10).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        self.temporal_unit = 0

    def _get_data(self, flag, downstream=False):
        args = self.args

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 2

        if flag in ('test', 'val') or downstream:
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.batch_size;
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False;
            drop_last = False;
            batch_size = 1;
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq

        if flag == 'train' and not downstream:
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[3000, 0, 0],
                features=args.features,
                target=args.target,
                inverse=args.inverse,
                timeenc=timeenc,
                freq=freq,
                cols=args.cols
            )
        else:
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                inverse=args.inverse,
                timeenc=timeenc,
                freq=freq,
                cols=args.cols
            )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train', downstream=False)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.net.train()
            epoch_time = time.time()
            for i, (batch_x, _, batch_x_mark, _) in enumerate(train_loader):
                iter_count += 1

                x = torch.cat([batch_x.float(), batch_x_mark.float()], dim=-1).to(self.device)

                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

                optimizer.zero_grad()

                out1 = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:]

                out2 = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l]

                loss = hierarchical_contrastive_loss(
                    out1,
                    out2,
                    temporal_unit=self.temporal_unit
                )

                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)

                train_loss.append(loss.item())
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))

                
    def test(self, setting):
        _, train_loader = self._get_data(flag='train', downstream=True)
        _, vali_loader = self._get_data(flag='val', downstream=True)
        _, test_loader = self._get_data(flag='test', downstream=True)
        
        self.net.eval()

        train_repr = []
        valid_repr = []
        test_repr = []
        
        train_data = []
        valid_data = []
        test_data = []

        for i, (batch_x, batch_y, batch_x_mark, _) in enumerate(train_loader):
            x = torch.cat([batch_x_mark.float(), batch_x.float()], dim=-1)
            x = x.to(self.device)
            out = self.net(x)[:, -1].detach().cpu()
            train_repr.append(out)
            train_data.append(batch_y.view(batch_y.size(0), -1))
            
        for i, (batch_x, batch_y, batch_x_mark, _) in enumerate(vali_loader):
            x = torch.cat([batch_x_mark.float(), batch_x.float()], dim=-1)
            x = x.to(self.device)
            out = self.net(x)[:, -1].detach().cpu()
            valid_repr.append(out)
            valid_data.append(batch_y.view(batch_y.size(0), -1))
            
        for i, (batch_x, batch_y, batch_x_mark, _) in enumerate(test_loader):
            x = torch.cat([batch_x_mark.float(), batch_x.float()], dim=-1)
            x = x.to(self.device)
            out = self.net(x)[:, -1].detach().cpu()
            test_repr.append(out)
            test_data.append(batch_y.view(batch_y.size(0), -1))
            
        train_repr = torch.cat(train_repr, dim=0).numpy()
        train_data = torch.cat(train_data, dim=0).numpy()
        valid_repr = torch.cat(valid_repr, dim=0).numpy()
        valid_data = torch.cat(valid_data, dim=0).numpy()
        test_repr = torch.cat(test_repr, dim=0).numpy()
        test_data = torch.cat(test_data, dim=0).numpy()

        lr = fit_ridge(train_repr, train_data, valid_repr, valid_data, MAX_SAMPLES=100000)
        test_pred = lr.predict(test_repr)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mae, mse, rmse, mape, mspe = metric(test_pred, test_data)

        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))

        return