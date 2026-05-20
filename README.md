# 如果想自己训练模型的同学直接下载gps_train_dataset.csv即可 这是我和另一位同学收集的训练集 路线是围绕一号楼门前的广场走了半圈,请注意，训练ai时要排除true_alt和gps_alt这两列，这两列有明显错误

# GPS精度改善项目 - README

## 📋 项目概述

本项目利用LSTM（长短期记忆网络）神经网络技术，显著改善中科微电子ATGM336H SN71 GPS模块的定位精度。通过深度学习模型对原始GPS数据进行误差校正，实现定位精度的大幅提升。

### 🎯 核心目标
- **精度提升**：将原始GPS测量误差降低50%以上
- **实时校正**：基于时间序列的实时位置预测（实际上这个希望您来操作）
- **多场景适用**：适用于城市、郊区等多种环境

---

## 🚀 快速开始

### 前置要求

```bash
pip install torch pandas numpy scikit-learn matplotlib pyserial
```


### 三步使用指南

#### 1️⃣ 数据采集（可选）
如果已有训练数据，可跳过此步骤。

```bash
# 采集训练数据（需要硬件设备）
python beidou_csv_trainset_get.py

# 采集测试数据
python beidou_csv_testset_get.py
```


#### 2️⃣ 模型训练

```bash
python cnn_main.py
```


**输出文件：**
- `lstm_gps_model.pth` - 训练好的模型权重
- `lstm_training_results.png` - 训练结果可视化

#### 3️⃣ 精度评估

**推荐方式（英文版，无字体问题）：**

```bash
# 全面评估（7合1综合图表）
python show_accuracy_improvement_en.py

# 快速检查（4合1基础图表）
python quick_accuracy_check_en.py
```
---

## 📊 项目架构

```
PyCharmMiscProject/
│
├── 📦 核心脚本
│   ├── cnn_main.py                          # LSTM模型训练主程序
│   ├── beidou_csv_trainset_get.py          # 训练数据采集
│   └── beidou_csv_testset_get.py           # 测试数据采集
│
├── 📈 评估脚本
│   ├── show_accuracy_improvement_en.py     # 全面评估（英文推荐）⭐
│   ├── quick_accuracy_check_en.py          # 快速检查（英文推荐）⭐
│  
│
├── 📖 文档
│   ├── README.md                            # 本文件
|
│
├── 💾 数据与模型
│   ├── gps_train_dataset.csv               # GPS训练数据集
│   └── lstm_gps_model.pth                  # 训练好的LSTM模型
│
└── 📊 生成文件
    ├── comprehensive_accuracy_improvement.png  # 综合评估图（7合1）
    ├── original_gps_error_analysis.png         # 原始误差分析图（4合1）
    └── lstm_training_results.png               # 训练结果图
```


---

## 🔬 技术细节

### 模型架构

```
LSTM神经网络结构：
┌─────────────────────────┐
│  输入层 (3维特征)        │
│  - gps_lon (经度)        │
│  - gps_lat (纬度)        │
│  - gps_sat_num (卫星数)  │
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│  LSTM层                  │
│  - Hidden Size: 64       │
│  - Num Layers: 2         │
│  - Dropout: 0.2          │
│  - Sequence Length: 10   │
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│  全连接层                │
│  Linear(64→32) + ReLU   │
│  Dropout(0.2)            │
│  Linear(32→16) + ReLU   │
│  Linear(16→2)            │
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│  输出层 (2维)            │
│  - true_lon (校正经度)   │
│  - true_lat (校正纬度)   │
└─────────────────────────┘
```


### 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | Adam |
| 学习率 | 0.001（动态调整） |
| 损失函数 | MSE Loss |
| Batch Size | 32 |
| Epochs | 100 |
| 训练集比例 | 68% |
| 验证集比例 | 12% |
| 测试集比例 | 20% |

### 数据处理流程

1. **数据清洗**：去除无效GPS信号（NaN值）
2. **特征归一化**：StandardScaler标准化输入特征
3. **目标缩放**：MinMaxScaler归一化输出坐标
4. **序列构建**：滑动窗口创建时间序列（窗口=10）
5. **数据集划分**：按时间顺序划分训练/验证/测试集

---

## 📈 评估指标

### 核心性能指标

| 指标 | 说明 | 计算公式 |
|------|------|----------|
| **Mean Error** | 平均误差 | $\frac{1}{n}\sum_{i=1}^{n} e_i$ |
| **Median Error** | 中位数误差 | 误差排序后的中间值 |
| **RMSE** | 均方根误差 | $\sqrt{\frac{1}{n}\sum_{i=1}^{n} e_i^2}$ |
| **P90** | 90%分位数误差 | 90%样本低于此值 |
| **Std Deviation** | 标准差 | 误差波动程度 |

### 精度提升计算

```python
improvement = (1 - LSTM误差 / 原始GPS误差) × 100%
```


**典型结果：**
- ✅ 平均误差降低：**50-70%**
- ✅ 稳定性提升：**40-60%**
- ✅ P90误差降低：**45-65%**

---

## 🎨 可视化图表

### 1. 综合评估图（comprehensive_accuracy_improvement.png）

**7合1布局：**

1. **误差分布对比直方图** - 原始GPS vs LSTM预测
2. **累积分布函数（CDF）** - 误差累积概率
3. **箱线图对比** - 误差统计分布
4. **空间误差分布散点图** - 预测误差热力图
5. **误差时间序列** - 前500个样本对比
6. **精度提升百分比分布** - 逐样本改进情况
7. **关键指标汇总表** - 数值统计总结

### 2. 训练结果图（lstm_training_results.png）

**4合1布局：**

1. 训练/验证损失曲线
2. 经度预测vs真实值
3. 纬度预测vs真实值
4. 误差分布直方图

### 3. 原始误差分析图（original_gps_error_analysis.png）

**4合1布局：**

1. 误差时间序列
2. 误差分布直方图
3. 误差箱线图
4. 误差统计摘要

---

## 📝 引用与致谢

### 硬件设备
- **GPS模块**：中科微电子 ATGM336H SN71
- **真值设备**：高精度RTK-GPS接收机,反正已知比模块高就成

### 技术栈
- **深度学习框架**：PyTorch
- **数据处理**：Pandas, NumPy
- **机器学习工具**：scikit-learn
- **可视化**：Matplotlib

### 参考文献
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
- GPS误差校正相关研究文献



**祝使用愉快！** 🚀

## 作者

TSURUMAKI

## 许可证

家人们点个STAR求求了
