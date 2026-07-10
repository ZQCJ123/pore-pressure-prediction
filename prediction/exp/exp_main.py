from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, SCformer, Preformer, Preformer_wo_MS, Decom_FullAttention, Decom_ProbAttention, Decom_LogAttention, SCformer_de2, LogTrans, lstm
from utils.tools import EarlyStopping, adjust_learning_rate, visual,  visual_more, visual_all, Regression
from utils.metrics import metric
import joblib
from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pandas
import os
import time
from scipy import stats
import warnings
import matplotlib.pyplot as plt
# from pyemd import EMD, EEMD, CEEMDAN  # CEEMDAN # pip install EMD-signal
# from sampen import sampen2  # Sample Entropy
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
plt.rcParams.update({'font.size': 20})


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Preformer': Preformer,
            'Preformer_wo_MS': Preformer_wo_MS,
            'SCformer': SCformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'LogTrans': LogTrans,
            'Decom_Log': Decom_LogAttention,
            'Decom_Prob': Decom_ProbAttention,
            'Decom_Full': Decom_FullAttention,
            'SCformer_de2': SCformer_de2,
            'LSTM': lstm,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def criterion(self, predict, true, rms_g_e, return_components=False):
        mse_loss = nn.MSELoss()(predict, true)
        gradient_constraint = nn.MSELoss()(predict, rms_g_e)
        total_loss = 0.7 * mse_loss + 0.3 * gradient_constraint

        if return_components:
            return total_loss, mse_loss, gradient_constraint

        return total_loss

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            # ppg_model = load_model('./pre model/vsp2pp-lstm1.h')
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                # rms_g_e = batch_y[:, -self.args.pred_len:, -2:-1].detach().cpu()
                # batch_x = torch.cat((batch_x[:, :, :-2], batch_x[:, :, -1:]), dim=2)
                # batch_y = torch.cat((batch_y[:, :, :-2], batch_y[:, :, -1:]), dim=2)
                # pre_dtco_in = torch.cat([batch_x[:, :, 0:6], batch_x[:, :, 10:11]], dim=2)
                # pre_rho8_in = torch.cat([batch_x[:, :, 0:1], batch_x[:, :, 6:11]], dim=2)
                # pre_res_in = torch.cat([batch_x[:, :, 0:1], batch_x[:, :, 10:11], batch_x[:, :, 12:17]], dim=2)
                # pre_ppg_in = batch_x[:, -self.args.pred_len:, 0:6]
                # dtco_model = joblib.load('./pre model/dtco_adaboost_model.joblib')
                # pre_dtco_out = dtco_model.predict(pre_dtco_in.view(-1, pre_dtco_in.size(-1)).numpy())
                # rho8_model = joblib.load('./pre model/rho8_RF_model.joblib')
                # pre_rho8_out = rho8_model.predict(pre_rho8_in.view(-1, pre_rho8_in.size(-1)).numpy())
                # res_model = joblib.load('./pre model/res_xgb_model.joblib')
                # pre_res_out = res_model.predict(pre_res_in.view(-1, pre_res_in.size(-1)).numpy())
                # ppg_model = load_model('./pre model/vsp2pp-lstm.h')
                # pre_ppg_out = np.array(ppg_model.predict(pre_ppg_in.numpy(), verbose=0))
                # pre_ppg_out = torch.from_numpy(np.array([])).to(self.device)
                pre_ppg_out = batch_y[:, -self.args.pred_len:, -2:-1]

                # batch_x_new = np.concatenate((batch_x.numpy(), pre_dtco_out.reshape(-1, self.args.seq_len, 1),
                #                               pre_rho8_out.reshape(-1, self.args.seq_len, 1),
                #                               pre_res_out.reshape(-1, self.args.seq_len, 1)), axis=2)
                # batch_x = torch.from_numpy(batch_x_new).to(self.device)
                # pre_ppg_out = torch.from_numpy(pre_ppg_out).to(self.device)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            outputs = self.model(batch_x, dec_inp)[0]
                        else:
                            # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            outputs = self.model(batch_x, dec_inp)
                else:
                    if self.args.output_attention:
                        # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        outputs = self.model(batch_x, dec_inp)[0]
                    else:
                        # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = self.model(batch_x, dec_inp, batch_y)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                pred = outputs[:, :, f_dim:].detach().cpu()
                true = batch_y.detach().cpu()

                loss = self.criterion(pred, true, pre_ppg_out.float())

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        # criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        loss_record = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            epoch_mse_loss = []
            epoch_gradient_constraint = []
            epoch_total_loss = []

            self.model.train()
            epoch_time = time.time()
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            # ppg_model = load_model('./pre model/vsp2pp-lstm1.h')
            for i, (batch_x, batch_y) in enumerate(train_loader):
                # rms_g_e = batch_y[:, -self.args.pred_len:, -2:-1].to(self.device)
                # batch_x = batch_x.numpy()
                # pre_dtco_in = torch.cat([batch_x[:, :, 0:6], batch_x[:, :, 10:11]], dim=2)
                # pre_rho8_in = torch.cat([batch_x[:, :, 0:1], batch_x[:, :, 6:11]], dim=2)
                # pre_res_in = torch.cat([batch_x[:, :, 0:1], batch_x[:, :, 10:11], batch_x[:, :, 12:17]], dim=2)
                # pre_ppg_in = batch_x[:, -self.args.pred_len:, 0:6]
                # dtco_model = joblib.load('./pre model/dtco_adaboost_model.joblib')
                # pre_dtco_out = dtco_model.predict(pre_dtco_in.view(-1, pre_dtco_in.size(-1)).numpy())
                # rho8_model = joblib.load('./pre model/rho8_RF_model.joblib')
                # pre_rho8_out = rho8_model.predict(pre_rho8_in.view(-1, pre_rho8_in.size(-1)).numpy())
                # res_model = joblib.load('./pre model/res_xgb_model.joblib')
                # pre_res_out = res_model.predict(pre_res_in.view(-1, pre_res_in.size(-1)).numpy())
                # ppg_model = load_model('./pre model/vsp2pp-lstm.h')
                # pre_ppg_out = np.array(ppg_model.predict(pre_ppg_in.numpy(), verbose=0))
                # pre_ppg_out = torch.from_numpy(np.array([])).to(self.device)
                pre_ppg_out = batch_y[:, -self.args.pred_len:, -2:-1].to(self.device)

                # batch_x_new = np.concatenate((batch_x.numpy(), pre_dtco_out.reshape(-1, self.args.seq_len, 1), pre_rho8_out.reshape(-1, self.args.seq_len, 1), pre_res_out.reshape(-1, self.args.seq_len, 1)), axis=2)
                # batch_x = torch.from_numpy(batch_x_new).to(self.device)
                # pre_ppg_out = torch.from_numpy(pre_ppg_out).to(self.device)

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # print(batch_x.shape)
                # print(batch_x_mark.shape)
                # print(dec_inp.shape)
                # print(batch_y_mark.shape)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            outputs = self.model(batch_x, dec_inp)[0]
                        else:
                            # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            outputs = self.model(batch_x, dec_inp)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = self.criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        outputs = self.model(batch_x, dec_inp)[0]
                    else:
                        # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                        outputs = self.model(batch_x, dec_inp, batch_y)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    # batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    loss, mse_loss, gradient_constraint = self.criterion(
                        outputs[:, :, f_dim:],
                        batch_y,
                        pre_ppg_out.float(),
                        return_components=True
                    )

                    train_loss.append(loss.item())
                    epoch_mse_loss.append(mse_loss.item())
                    epoch_gradient_constraint.append(gradient_constraint.item())
                    epoch_total_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, self.criterion)
            test_loss = self.vali(test_data, test_loader, self.criterion)
            loss_record.append({
                'epoch': epoch + 1,
                'mse_loss': np.average(epoch_mse_loss),
                'gradient_constraint': np.average(epoch_gradient_constraint),
                'total_loss': np.average(epoch_total_loss)
            })

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        loss_df = pandas.DataFrame(loss_record)
        loss_df.to_csv('./loss.csv', index=False)
        print("Loss components saved")

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        all_mse_min = 10000
        all_index_min = 0

        # 用于记录每次预测耗时
        time_records = []

        with torch.no_grad():
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            # ppg_model = load_model('./pre model/vsp2pp-lstm1.h')
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                # 只统计模型前向预测耗时，不包含反归一化、画图、指标计算等后处理时间
                if torch.cuda.is_available() and 'cuda' in str(self.device):
                    torch.cuda.synchronize()

                start_time = time.perf_counter()

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, dec_inp)[0]
                        else:
                            outputs = self.model(batch_x, dec_inp)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, dec_inp)[0]
                    else:
                        outputs = self.model(batch_x, dec_inp, batch_y)

                if torch.cuda.is_available() and 'cuda' in str(self.device):
                    torch.cuda.synchronize()

                end_time = time.perf_counter()

                batch_time_s = end_time - start_time
                batch_size_now = batch_x.shape[0]
                avg_time_per_sample_s = batch_time_s / batch_size_now

                time_records.append({
                    'batch_index': i,
                    'batch_size': int(batch_size_now),
                    'batch_time_s': batch_time_s,
                    'batch_time_ms': batch_time_s * 1000,
                    'avg_time_per_sample_s': avg_time_per_sample_s,
                    'avg_time_per_sample_ms': avg_time_per_sample_s * 1000
                })

                f_dim = -1 if self.args.features == 'MS' else 0
                # 创建一个形状为 (32, 56, 1) 的全零张量
                zeros = torch.zeros(outputs.shape[0], outputs.shape[1], self.args.enc_in).to(self.device)
                # 使用切片在倒数第二列插入全零列
                outputs = torch.cat((zeros, outputs), dim=2)
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                # batch_y = torch.cat((zeros, batch_y), dim=2)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                batch_y = batch_y.reshape(batch_y.shape[0]*batch_y.shape[1], batch_y.shape[2])
                batch_y = test_data.scaler.inverse_transform(batch_y)
                batch_y = batch_y.reshape(self.args.batch_size, self.args.pred_len, self.args.enc_in+1)
                outputs = outputs.reshape(outputs.shape[0]*outputs.shape[1], outputs.shape[2])
                outputs = test_data.scaler.inverse_transform(outputs)
                outputs = outputs.reshape(self.args.batch_size, self.args.pred_len, self.args.enc_in+1)

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 1 == 0:
                    # print(i)
                    mse_min = 10000
                    index_min = 0
                    for k in range(true.shape[0]):
                        current_mse = np.mean((pred[k, :, -1] - true[k, :, -1]) ** 2)
                        if current_mse < mse_min:
                            index_min = k
                            mse_min = current_mse
                    print(i, index_min)
                    if mse_min < all_mse_min:
                        all_mse_min = mse_min
                        all_index_min = i
                    zeros1 = torch.zeros(batch_x.shape[0], batch_x.shape[1], 1).to(self.device)
                    batch_x = torch.cat((zeros1, batch_x), dim=2)
                    input = batch_x.detach().cpu().numpy()
                    input = input.reshape(input.shape[0] * input.shape[1], input.shape[2])
                    input = test_data.scaler.inverse_transform(input)
                    input = input.reshape(self.args.batch_size, self.args.seq_len, self.args.enc_in+1)
                    gt = np.concatenate((input[index_min, :, -1], true[index_min, :, -1]), axis=0)
                    pd = np.concatenate((input[index_min, :, -1], pred[index_min, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.jpg'))

        # 保存每次预测耗时
        time_df = pandas.DataFrame(time_records)

        if len(time_df) > 0:
            total_time_s = time_df['batch_time_s'].sum()
            total_samples = time_df['batch_size'].sum()
            avg_time_per_sample_s = total_time_s / total_samples

            summary_df = pandas.DataFrame([{
                'batch_index': 'TOTAL',
                'batch_size': int(total_samples),
                'batch_time_s': total_time_s,
                'batch_time_ms': total_time_s * 1000,
                'avg_time_per_sample_s': avg_time_per_sample_s,
                'avg_time_per_sample_ms': avg_time_per_sample_s * 1000
            }])

            time_df = pandas.concat([time_df, summary_df], ignore_index=True)

        time_csv_path = os.path.join(folder_path, 'prediction_time.csv')
        # time_df.to_csv(time_csv_path, index=False, encoding='utf-8-sig')
        # print(f'预测耗时已保存到: {time_csv_path}')

        print("最小mse对应的i为" + str(all_index_min))
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        preds = preds[:, :, -1]
        trues = trues[:, :, -1]
        print('test shape:', preds.shape, trues.shape)
        Preds = np.array([])
        Trues = np.array([])
        for index in range(0, preds.shape[0], self.args.pred_len):
            Preds = np.append(Preds, preds[index, :])
            Trues = np.append(Trues, trues[index, :])

        # # 将两个数组按列合并
        merged = np.column_stack((Preds, Trues))
        # 将合并后的数组放入 DataFrame
        df = pandas.DataFrame(merged, columns=['Preds', 'Trues'])
        df.to_csv('Men_Transformer_alldata.csv', index=False)

        visual_more(Trues, Preds, os.path.join(self.args.root_path, self.args.data_path), os.path.join(folder_path, 'all predict result.jpg'))
        Regression(Trues, Preds, os.path.join(folder_path, 'linear regression analysis.jpg'))
        visual_all(Trues, Preds, os.path.join(self.args.root_path, self.args.data_path), os.path.join(folder_path, 'all result.jpg'))

        mae, mse, rmse, mape, mspe, IA, R, PICP, MPIW = metric(Preds, Trues)
        print('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}, IA:{}, R:{}, PICP:{}, MPIW:{}'.format(mae, mse, rmse, mape, mspe, IA, R, PICP, MPIW))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}, IA:{}, R:{}, PICP:{}, MPIW:{}'.format(mae, mse, rmse, mape, mspe, IA, R, PICP, MPIW))
        f.write('\n')
        f.write('\n')
        f.close()
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, dec_inp)[0]
                        else:
                            outputs = self.model(batch_x, dec_inp)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, dec_inp)[0]
                    else:
                        outputs = self.model(batch_x, dec_inp)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
