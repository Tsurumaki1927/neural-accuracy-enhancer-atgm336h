# 如果想自己训练模型的同学直接下载gps_train_dataset.csv即可 这是我和另一位同学收集的训练集 路线是围绕一号楼门前的广场走了半圈

# GPS 精度改善 - 基于 LSTM 神经网络

## 项目简介

本项目旨在利用深度学习技术（LSTM 神经网络）来提高中科微电子 ATGM336H SN71 GPS 模块的定位精度。通过收集 GPS 原始数据和真实位置数据作为训练集，训练一个 LSTM 模型来预测更准确的位置信息。

## 功能特点

- 实时采集 GPS 模块数据和手机真值数据
- 数据预处理和时序序列构建
- 基于 LSTM 的神经网络模型训练
- 模型评估与可视化
- 精度提升效果对比分析

## 项目结构

```

├── beidou_csv_trainset_get.py    # 训练数据采集脚本
├── beidou_csv_testset_get.py     # 测试数据采集脚本
├── cnn_main.py                   # 主程序（LSTM模型训练和评估）
├── gps_train_dataset.csv         # 训练数据集
├── lstm_gps_model.pth            # 训练好的模型文件
├── lstm_training_results.png     # 训练结果可视化图
└── README.md                     # 项目说明文档
```
## 环境依赖

- Python 3.7+
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib
- pyserial

安装依赖：
```
bash
pip install torch pandas numpy scikit-learn matplotlib pyserial
```
## 使用方法

### 1. 数据采集

#### 训练数据采集
运行 `beidou_csv_trainset_get.py` 来采集训练数据：

```
bash
python beidou_csv_trainset_get.py
```
该脚本会：
- 通过 UDP 接收手机发送的真实位置数据
- 从串口读取 GPS 模块的 NMEA 数据
- 将时间对齐的数据对保存到 `gps_train_dataset.csv
**注意修改端口和内网ip**

#### 测试数据采集
运行 `beidou_csv_testset_get.py` 来采集测试数据：

```
bash
python beidou_csv_testset_get.py
```
### 2. 模型训练

运行 `cnn_main.py` 进行模型训练和评估：

```
bash
python cnn_main.py
```
该脚本会：
- 加载并预处理数据
- 创建时序序列
- 构建 LSTM 模型
- 训练模型
- 在测试集上评估性能
- 保存训练好的模型
- 生成可视化结果图

## 模型架构

使用的 LSTM 模型包含：
- 输入层：3 个特征（经度、纬度、卫星数量）
- 2 层 LSTM 隐藏层（64 个隐藏单元）
- 全连接输出层：2 个输出（预测的经度、纬度）
- Dropout 正则化防止过拟合

## 数据格式

训练数据文件 `gps_train_dataset.csv` 包含以下列：
- `timestamp`: 时间戳
- `gps_lon`: GPS 模块测量的经度
- `gps_lat`: GPS 模块测量的纬度
- `gps_alt`: GPS 模块测量的海拔
- `gps_sat_num`: 可见卫星数量
- `true_lon`: 真实经度（来自手机）
- `true_lat`: 真实纬度（来自手机）
- `true_alt`: 真实海拔（来自手机）

## 结果展示

训练完成后会自动生成 `lstm_training_results.png` 图片，包含：
1. 训练和验证损失曲线
2. 经度预测 vs 真实值散点图
3. 纬度预测 vs 真实值散点图
4. 预测误差分布直方图

## 性能指标

模型评估时会显示以下指标：
- 平均经度误差（微度和米）
- 平均纬度误差（微度和米）
- 平均总误差（米）
- 最大/最小误差（米）
- RMS 误差（米）
- 中位数误差（米）
- 与原始 GPS 数据的精度提升百分比

## 配置参数

在 `cnn_main.py` 中可以调整以下超参数：
- `SEQ_LENGTH`: 序列长度（默认 10）
- `HIDDEN_SIZE`: LSTM 隐藏层大小（默认 64）
- `NUM_LAYERS`: LSTM 层数（默认 2）
- `BATCH_SIZE`: 批次大小（默认 32）
- `EPOCHS`: 训练轮数（默认 100）
- `LEARNING_RATE`: 学习率（默认 0.001）
- `DROPOUT`: Dropout 比率（默认 0.2）

## 注意事项

1. 确保 GPS 模块正确连接到计算机串口
2. 手机需要通过网络向指定 IP 和端口发送位置数据
3. 数据采集过程中请保持设备稳定移动以获得更好的训练数据
4. 根据实际硬件配置可能需要调整串口参数和网络设置

## 作者

TSURUMAKI

## 许可证

家人们点个STAR求求了
