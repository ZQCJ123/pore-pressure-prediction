import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
from scipy import stats

plt.switch_backend('agg')
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
plt.rcParams.update({'font.size': 20})


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'const':
        lr_adjust = {epoch: args.learning_rate}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure(figsize=(16, 9))
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.xlabel("step")
    plt.ylabel('Pore Pressure (MPa)')
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def visual_one(true, preds=None, name='./pic/test.pdf', l=0, h=1, d=0.1):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    y = range(l, h, d)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.yticks(y)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def visual_more(true, preds=None, datapath=None, name='./pic/test.pdf', l=0, h=1, d=0.1):
    """
    Results visualization
    """
    df_raw = pd.read_csv('./data/ETT/wellA_pr_all.csv')
    depth = df_raw['TDEP'].values
    num_train = int(len(df_raw) * 0.7)
    num_test = int(len(df_raw) * 0.2)
    num_vali = len(df_raw) - num_train - num_test
    num_train = 0
    num_vali = 0
    plt.figure(figsize=(9, 16))
    plt.plot(true, depth[num_train + num_vali:num_train + num_vali + true.shape[0]], label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, depth[num_train + num_vali:num_train + num_vali + preds.shape[0]], label='Prediction',
                 linewidth=1)
    ax = plt.gca()  # 获取到当前坐标轴信息
    ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
    ax.invert_yaxis()
    # 设置X轴标签和它的位置
    ax.set_xlabel('Pore Pressure (MPa)', labelpad=10)
    ax.xaxis.set_label_position('top')  # 将X轴标签移到上方
    plt.ylabel('Depth (mbsf)')
    plt.title('Pore Pressure Prediction', pad=10)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


# def visual_attention(true, scores, name='./pic/attn.pdf'):
#     """
#     true: L (past + future)
#     scores:
#     """
#     plt.figure()
#     plt.subplot(221)
#
#     plt.subplot(222)
#     plt.plot
#
#     plt.subplot(212)
#     plt.plot(true, label='GroundTruth', linewidth=2)
#     plt.savefig(name, bbox_inches='tight')


def visual_all(true, preds=None, datapath=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    df_raw = pd.read_csv('./data/ETT/wellA_pr_all.csv')
    depth = df_raw['TDEP'].values
    index = np.array([])
    label = np.array([])
    for i in range(0, depth.shape[0], 300):
        index = np.append(index, i)
        label = np.append(label, round(depth[i]))
    label = label.astype(int)
    num_train = int(len(df_raw) * 0.7)
    num_test = int(len(df_raw) * 0.2)
    num_vali = len(df_raw) - num_train - num_test
    Preds = np.append(df_raw.iloc[:num_train + num_vali, -1], preds)
    Trues = np.append(df_raw.iloc[:num_train + num_vali, -1], true)
    Preds = preds
    Trues = true
    plt.figure(figsize=(9, 16))
    plt.plot(Trues, depth[:Trues.shape[0]], label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(Preds, depth[:Trues.shape[0]], label='Prediction', linewidth=1)
    ax = plt.gca()  # 获取到当前坐标轴信息
    ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
    ax.invert_yaxis()
    # 设置X轴标签和它的位置
    ax.set_xlabel('Pore Pressure (MPa)', labelpad=10)
    ax.xaxis.set_label_position('top')  # 将X轴标签移到上方
    plt.ylabel('Depth (mbsf)')
    plt.title('Pore Pressure Prediction', pad=10)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def Regression(true, preds=None, name='./pic/test.pdf'):
    # 进行线性回归分析，得到斜率，截距，相关系数等参数
    slope, intercept, r_value, p_value, std_err = stats.linregress(true, preds)
    # 打印回归方程和相关系数
    print(f"Regression: R={r_value:.3f}: Output={slope:.3f}*Target+{intercept:.3f}")
    # 绘制散点图和回归线
    plt.figure()
    plt.scatter(true, preds, color='#818181', label="Data")
    plt.plot(true, slope * true + intercept, color="black", label="Fit")
    plt.xlabel("true")
    plt.ylabel("preds")
    plt.title(f"R={r_value:.3f}: preds~={slope:.3f}*true+{intercept:.3f}")
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
