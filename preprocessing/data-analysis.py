import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import chardet
from scipy.interpolate import interp1d
import math
import scipy
import pywt

plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
plt.rcParams.update({'font.size': 15})
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "text.usetex": True,  # 启用 LaTeX
    "font.family": "serif",  # 使用 LaTeX 的罗马字体
})


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

####################################################数据分析#############################################################
# # 读取CSV文件
# file_path = './Nimblefoot-1.csv'  # ✅ 请替换成你的文件路径
# data = pd.read_csv(file_path)
# # 仅保留数值型列（避免 object 类型报错）
# numeric_data = data.select_dtypes(include=['number'])
# # 计算统计指标：Min, Mean, Max, Std
# summary_stats = pd.DataFrame({
#     'Min': numeric_data.min(),
#     'Mean': numeric_data.mean(),
#     'Max': numeric_data.max(),
#     'Std': numeric_data.std()
# })
# # 打印结果
# print(summary_stats)
# # 可选：保存为CSV
# summary_stats.to_csv("./wellC_stats.csv")
############################################## 密度声波交汇图 #############################################################
# df = pd.read_csv('Nimblefoot-1-prep-data.csv')
# split_idx1 = int(len(df) * 0)
# split_idx2 = int(len(df) * 1)
# df = df.iloc[split_idx1:split_idx2]
# data = pd.DataFrame({
#     'depth': np.array(df['DEPT']).reshape(-1),
#     'measurement1': np.array(df['RHO8']).reshape(-1),
#     'measurement2': np.array(df['DTCO']).reshape(-1)
# })
# # 提取深度数据和测量数据
# depth = data['depth']
# measurement1 = data['measurement1']
# measurement2 = 1000000/data['measurement2']
# # 创建颜色映射
# # 使用深度数据生成一个颜色梯度
# cmap = plt.get_cmap('viridis')
# normalize = plt.Normalize(vmin=depth.min(), vmax=depth.max())
# colors = cmap(normalize(depth))
# # 绘制散点图
# plt.figure(figsize=(10, 6))
# scatter = plt.scatter(measurement1, measurement2, c=colors, s=10, cmap='viridis')
# # 添加颜色条
# cbar = plt.colorbar(scatter)
# cbar.set_label('Depth(m)')
# # 假设 cb 是您的 colorbar 对象
# cbar.mappable.set_clim(vmin=depth.min(), vmax=depth.max())
# # 设置图形标题和标签
# plt.title('Torosa-3 (S4) Transit Time – Density Crossplot')
# plt.xlabel('Formation Density (g/$\mathrm{cm}^3$)')
# plt.ylabel('Transit Time (m/s)')
# plt.tight_layout()
# # plt.savefig('C:/Users/28578/Desktop/Lightfinger-1 Cross Plot Density and Vp.svg', dpi=300, bbox_inches='tight', pad_inches=0)
# # 显示图形
# plt.show()


############################################### 深度匹配 ################################################################
# # encoding1：匹配数据
# # encoding2：匹配深度
# encoding1 = detect_encoding('Torosa-6 Time vs depth listing_v2b.csv')
# encoding2 = detect_encoding('Torosa-6-data.csv')
# df1 = pd.read_csv('Torosa-6 Time vs depth listing_v2b.csv', encoding=encoding1)
# df2 = pd.read_csv('Torosa-6-data.csv', encoding=encoding2)
# # 使用插值法将钻井数据重采样以匹配测井数据的深度
# depths_to_interpolate = df2['DEPT']
# # 创建一个新的 DataFrame，包含重采样后的钻井数据
# interpolated_df = pd.DataFrame({'TDEP': depths_to_interpolate})
# # 对每一列进行插值
# for col in df1.columns[1:]:
#     interpolated_values = np.interp(depths_to_interpolate, df1['Measured Depth'], df1[col])
#     interpolated_df[col] = interpolated_values
# # 输出结果
# print(interpolated_df)
# interpolated_df.to_csv('Torosa-6-VSP.csv', index=False)

# # 读取地震数据和测井数据
# table_a = pd.read_csv('VSP.csv')  # 地震数据（Measured Depth）
# table_b = pd.read_csv('Nimblefoot-1.csv')  # 测井数据（TDEP）
# # 确保深度列为浮点型，并统一列名
# table_a['Measured'] = table_a['Measured'].astype(float)
# table_b['DEPT'] = table_b['DEPT'].astype(float)
# # 遍历地震表中所有要插值的列（排除 'Measured Depth'）
# seismic_depth = table_a['Measured'].values
# for col in table_a.columns:
#     if col == 'Measured Depth':
#         continue  # 跳过深度列
#     # 原始值
#     seismic_value = table_a[col].values
#     # 构造插值函数
#     interp_func = interp1d(
#         seismic_depth, seismic_value, kind='linear', fill_value='extrapolate'
#     )
#     # 在测井深度上进行插值
#     interpolated_values = interp_func(table_b['DEPT'].values)
#     # 添加到测井数据中，命名为 'Interp_<原列名>'
#     table_b[f'Interp_{col}'] = interpolated_values
# plt.plot(table_a['Measured'], table_a['Interval Velocity'], label='Original', linewidth=2)
# # 插值结果
# plt.plot(table_b['DEPT'], table_b['Interp_Interval Velocity'], label='Interpolated', linestyle='--')
# plt.gca().invert_yaxis()  # 深度向下
# plt.legend()
# plt.xlabel('Depth (m)')
# plt.ylabel('Interval Velocity (m/s)')
# plt.title('Comparison of Original vs Interpolated')
# plt.show()
# # 保存带插值结果的表
# table_b.to_csv('1c.csv', index=False)


#################################################读取las文件#############################################################
# filepath = 'ILith_2009519.txt'
# las = pd.read_csv(filepath, delimiter='\t')
# data = pd.DataFrame(columns=las.keys(), index=range(las[las.keys()[0]].shape[0]))
# for i in range(len(las.keys())):
#     data[data.columns[i]] = las[las.keys()[i]]
# print(data)
# data.to_csv("ILith_2009519.csv", index=False, encoding='utf-8')


#################################### Eaton method calculate pore pressure###############################################
# # calculate pressure on overlying rocks sigma_v
# df = pd.read_csv('Nimblefoot-1-prep-data.csv')
# depth1 = np.array(df['DEPT']).reshape(-1, 1)
# rhom = np.array(df['RHO8']).reshape(-1, 1)
# dtco = np.array(df['DTCO']).reshape(-1, 1)
# # nimblefoot-1
# measure_depth = [3078.56, 3156.95, 3231.46, 3280.08, 3301.37, 3351.86, 3362.36, 3393.82, 3402.39, 3439.41, 3450.08, 3465.86]
# measure_pp = [4490.3, 4601.4, 4705.4, 4775.0, 4807.8, 4878.8, 4891.9, 4937.2, 4953.2, 4997.5, 5011.6, 5033.2]
# measure_pp = [x * 6.895 * 0.001 for x in measure_pp]
# print(measure_pp)
# sigma_v = []
# g = 9.81
# # nimblefoot-1 (1.5 2.5 3130.1436 1115)
# pressure_v = (1.5 + 2.5) * (3130.1436 - 1115) * g / 2
# for i in range(1, depth1.shape[0]):
#     pressure_v = pressure_v + (rhom[i, 0] + rhom[i - 1, 0]) * (depth1[i, 0] - depth1[i - 1, 0]) * g / 2
#     sigma_v.append(pressure_v * 0.001)
# # calculate hydrostatic pressure
# P_h = []
# pressure_h = 0
# mud_density = 1.025
# for i in range(depth1.shape[0]):
#     pressure_h = mud_density * g * (depth1[i, 0])
#     P_h.append(pressure_h * 0.001)
# # calculate pore pressure
# pp = []
# pressure_p = 0
# for i in range(1, depth1.shape[0]):
#     pressure_p = float(sigma_v[i - 1] - (sigma_v[i - 1] - P_h[i]) * (
#             (math.exp((14870 - depth1[i, 0]) / 2604) / dtco[i, 0]) ** 0.5))
#     pp.append(pressure_p)
# # Nimblefoot 14870 2604
# test = pd.DataFrame(
#     {'TDEP': depth1[1:, 0], 'sigma_v': np.array(sigma_v), 'P_h': np.array(P_h[1:]), 'pore pressure': pp})
# test.to_csv("C:/Users/28578/Desktop/pore pressure.csv", index=False)
# plt.figure(figsize=(4, 8))
# plt.plot(pp[:], depth1[1:, 0], color="red", label='Pore Pressure')
# # plt.plot(P_h[:], depth1[:], color="b", label='Hydrostatic Pressure')
# # plt.plot(sigma_v[:], depth1[1:, 0], color='y', label='Overburden Pressure')
# plt.scatter(measure_pp, measure_depth, label='Measure Data', s=50, zorder=5)
# plt.xlabel('Pore Pressure of Well C(MPa)', labelpad=10)
# plt.ylabel('TDEP (m)')
# plt.legend(loc=1, fontsize=10)  # 指定legend的位置,类似象限的位置
# ax = plt.gca()  # 获取到当前坐标轴信息
# ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
# ax.xaxis.set_label_position('top')  # 将X轴标签移到上方
# ax.invert_yaxis()
# plt.tight_layout()
# # plt.savefig('C:/Users/28578/Desktop/Pore Pressure Data.png', dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()


########################################################NCT############################################################
# df = pd.read_csv('Guardian-1-prep-data.csv')
# depth1 = np.array(df['DEPT']).reshape(-1, 1)
# # depth1 = depth1[:1650]
# dtco = np.array(df['DTCO']).reshape(-1, 1)
# # dtco = dtco[:1650]
# x = dtco.reshape(-1)
# x1 = np.log(x)
# y = depth1.reshape(-1)
# # 使用polyfit方法来拟合,并选择多项式,这里先使用2次方程
# z1 = np.polyfit(x1, y, 1)
# z2 = np.array([-2000, 12900])
# # 使用poly1d方法获得多项式系数,按照阶数由高到低排列
# p1 = np.poly1d(z1)
# p2 = np.poly1d(z2)
# # 在屏幕上打印拟合多项式
# print(p1)
# # 求对应x的各项拟合函数值
# fx = p1(x1)
# fx1 = p2(x1)
# df = pd.DataFrame({'depth': depth1[:, 0], 'dtco': dtco[:, 0]})
# # 随机生成n个不重复的索引
# random_indices = np.random.choice(df.index, int(df.shape[0] * 1), replace=False)
# # 通过索引获取对应的行
# random_rows = df.loc[random_indices]
# # 绘制坐标系散点数据及拟合曲线图
# plt.figure(figsize=(7, 10))
# # plt.plot(x1, y, 'b', label='origin data')
# plt.scatter(np.log(random_rows['dtco'].values.reshape(-1)), random_rows['depth'].values.reshape(-1),
#             label='origin data', s=5)
# plt.plot(x1, fx, 'r', label='polyfit data')
# plt.plot(x1, fx1, 'y', label='polyfit data')
# plt.xlabel('ln(V_p) (us/ft)', labelpad=10)
# plt.ylabel('DEPT (m)')
# plt.title('NCT of Sonic Travel Time Data', pad=10)
# plt.legend(loc=4)  # 指定legend的位置,类似象限的位置
# ax = plt.gca()  # 获取到当前坐标轴信息
# ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
# ax.xaxis.set_label_position('top')  # 将X轴标签移到上方
# ax.invert_yaxis()
# plt.tight_layout()
# # plt.savefig('C:/Users/28578/Desktop/NCT of Sonic Travel Time Data.png', dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()


#########################################################岩性深度匹配#####################################################
# litho_df = pd.read_csv("NImblefoot-1_InterpretLith.csv")  # 包含 top_depth, bottom_depth, lithology
# log_df = pd.read_csv("Nimblefoot-1.csv")     # 包含 Depth, DTCO, ...
# # 1. 岩性 → Mohs硬度 映射表
# litho_to_mohs = {
#     "Arg Limestone": 2.5,
#     "Marl": 2.5,
#     "Coal": 2.5,
#     "Argillaceous Calcilutite": 3.0,
#     "Argillaceous Calalutite": 3.0,
#     "Argillaceous Sandstone": 4.5,
#     "Argillaceous Siltstone": 4.0,
#     "Calcareous Claystone": 2.5,
#     "Calcareous Sandstone": 4.0,
#     "Calcarenite": 3.5,
#     "Calcilutite": 3.0,
#     "Calcisiltite": 3.5,
#     "Chert": 7.0,
#     "Claystone": 2.0,
#     "Dolomite": 3.5,
#     "Glauconitic Claystone": 2.5,
#     "Limestone": 3.0,
#     "Sandstone": 6.0,
#     "Sandy Siltstone": 5.0,
#     "Siltstone": 4.5,
#     "Silty Claystone": 3.0,
#     "Silty Sandstone": 5.5,
#     "Volcanic": 6.5,
#     "Anhydrite": 3.5,
# }
# # 2. 添加岩性 & 硬度到测井数据
# depths = log_df['DEPT'].values
# lithology_at_depth = []
# mohs_at_depth = []
# for d in depths:
#     match = litho_df[(litho_df['top_depth'] <= d) & (litho_df['bottom_depth'] > d)]
#     if not match.empty:
#         lith = match.iloc[0]['lithology']
#         mohs = litho_to_mohs.get(lith, None)
#     else:
#         lith = 'Unknown'
#         mohs = None
#     lithology_at_depth.append(lith)
#     mohs_at_depth.append(mohs)
#
# # 3. 添加新列
# log_df['Lithology'] = lithology_at_depth
# log_df['Mohs_Hardness'] = mohs_at_depth
# # 保存新表格（可选）
# log_df.to_csv("log_with_lithology.csv", index=False)


######################################### 数据预处理：异常值剔除、滤波#######################################################
df = pd.read_csv('nimblefoot.csv')
columns = df.columns
df1 = pd.DataFrame(columns=columns)
df1[columns[0]] = df[columns[0]]
for i in range(1, df.shape[1]):
    depth = np.array(df.iloc[:, 0:1]).reshape(-1, 1)
    dtco = np.array(df.iloc[:, i:i + 1]).reshape(-1, 1)
    threshold = 3
    data0 = dtco.reshape(-1)
    mean0 = np.mean(data0)
    std0 = np.std(data0)
    cutoff0 = std0 * threshold
    lower_bound0 = mean0 - cutoff0
    upper_bound0 = mean0 + cutoff0
    filtered_data = []
    for x in range(data0.shape[0]):
        if lower_bound0 <= data0[x] <= upper_bound0:
            filtered_data.append(data0[x])
        else:
            filtered_data.append(None)
            # print(x)
    filtered_data = pd.DataFrame(filtered_data)
    filtered_data = filtered_data.interpolate(method='linear')
    data = filtered_data.iloc[:, 0].values
    filtered_data = []
    for x in range(data.shape[0]):
        if 5 < x < data.shape[0] - 5:
            data1 = data[x - 5:x + 5]
            mean = np.mean(data1)
            std = np.std(data1)
            cutoff = std * threshold
            lower_bound = mean - cutoff
            upper_bound = mean + cutoff
        else:
            if x < 5:
                data1 = data[0:10]
                mean = np.mean(data1)
                std = np.std(data1)
                cutoff = std * threshold
                lower_bound = mean - cutoff
                upper_bound = mean + cutoff
            else:
                data1 = data[data.shape[0] - 10:-1]
                mean = np.mean(data1)
                std = np.std(data1)
                cutoff = std * threshold
                lower_bound = mean - cutoff
                upper_bound = mean + cutoff
        if lower_bound <= data[x] <= upper_bound:
            filtered_data.append(data[x])
        else:
            filtered_data.append(None)
    filtered_data = pd.DataFrame(filtered_data)
    filtered_data = filtered_data.interpolate(method='linear')
    # pre_processing_data = pd.DataFrame({'depth': depth1[:, 0], 'dtco': np.array(filtered_data)[:, 0]})
    # pre_processing_data.to_csv("C:/Users/28578/Desktop/127.csv")
    # SG滤波
    # filtered_signal = scipy.signal.savgol_filter(np.array(filtered_data).reshape(filtered_data.shape[0]), 553, 3)

    # 小波分解
    wavelet = 'db4'
    level = 2
    coeffs = pywt.wavedec(filtered_data.values.ravel(), wavelet, level=level)
    cA2, cD2, cD1 = coeffs

    # 使用 Savitzky-Golay 滤波器对 cA2 平滑处理
    # 参数含义：window_length（奇数），polyorder（多项式阶数）
    length = len(cA2)
    polyorder = 3

    cA2_smooth = scipy.signal.savgol_filter(cA2, window_length=10, polyorder=polyorder)

    # 用平滑后的 cA2 替换
    coeffs_denoised = [cA2_smooth, cD2, cD1]
    filtered_signal = pywt.waverec(coeffs_denoised, wavelet)
    filtered_signal = filtered_signal[:len(filtered_data)]

    # 截断重构信号长度匹配
    # filtered_signal = filtered_signal[:len(filtered_data)]

    # dtco_pre_processing_data = pd.DataFrame({'depth': depth1[:, 0], 'dtco': data_smooth})
    # dtco_pre_processing_data.to_csv("C:/Users/28578/Desktop/33.csv")
    plt.figure(figsize=(7, 10))
    plt.plot(data0, depth, color="red", label='Origin Data')
    plt.plot(filtered_data, depth, color="blue", label='Outlier-removed Data')
    plt.plot(filtered_signal, depth, color="yellow", label='Smoothed Data')
    plt.xlabel('Spontaneous Potential Log(mv)', labelpad=10)
    plt.ylabel('Depth (mbsf)')
    plt.legend(loc=4)  # 指定legend的位置,类似象限的位置
    ax = plt.gca()  # 获取到当前坐标轴信息
    ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
    ax.xaxis.set_label_position('top')  # 将X轴标签移到上方
    ax.invert_yaxis()
    plt.tight_layout()
    filepath = 'C:/Users/28578/Desktop/'+str(i)+'b.pdf'
    plt.savefig(filepath, format="pdf")
    plt.show()
    df1[columns[i]] = filtered_signal
# df1.to_csv('Nimblefoot-1-prep-data.csv', index=False)



