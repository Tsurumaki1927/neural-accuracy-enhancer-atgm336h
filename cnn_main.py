import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class GPSDataset(Dataset):
    """GPS数据集"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    """LSTM神经网络模型"""
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=2, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # LSTM前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])

        return out

def load_and_preprocess_data(csv_path):
    """加载和预处理数据"""
    print("正在加载数据...")
    df = pd.read_csv(csv_path)

    # 删除包含NaN的行（true_lat或true_lon为NaN的）
    df = df.dropna(subset=['true_lon', 'true_lat'])

    print(f"有效数据量: {len(df)} 条")

    # 特征选择（只使用经纬度和卫星数量）
    features = ['gps_lon', 'gps_lat', 'gps_sat_num']
    targets = ['true_lon', 'true_lat']

    # 提取特征和目标
    X = df[features].values
    y = df[targets].values

    # 归一化
    feature_scaler = StandardScaler()
    target_scaler = MinMaxScaler()

    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y)

    return X_scaled, y_scaled, feature_scaler, target_scaler, df

def create_sequences(X, y, seq_length=10):
    """创建时序序列"""
    X_seq, y_seq = [], []

    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length - 1])

    return np.array(X_seq), np.array(y_seq)

def calculate_haversine_loss(pred_lon, pred_lat, true_lon, true_lat):
    """计算Haversine距离损失（米）"""
    R = 6371000  # 地球半径（米）

    lon_diff = torch.radians(true_lon - pred_lon)
    lat_diff = torch.radians(true_lat - pred_lat)

    a = torch.sin(lat_diff/2)**2 + \
        torch.cos(torch.radians(pred_lat)) * torch.cos(torch.radians(true_lat)) * \
        torch.sin(lon_diff/2)**2

    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))

    return R * c

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    print(f"\n开始训练，设备: {device}")
    print("="*60)

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

    print("="*60)
    print(f"训练完成！最佳验证损失: {best_val_loss:.6f}")

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses

def evaluate_model(model, test_X, test_y, target_scaler, feature_scaler):
    """评估模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    with torch.no_grad():
        test_tensor = torch.FloatTensor(test_X).to(device)
        pred_scaled = model(test_tensor).cpu().numpy()

    # 反归一化
        pred = target_scaler.inverse_transform(pred_scaled)
        true = target_scaler.inverse_transform(test_y)

    # 计算误差（度）
    lon_error = np.abs(pred[:, 0] - true[:, 0])
    lat_error = np.abs(pred[:, 1] - true[:, 1])

    # 转换为米（近似）
    lon_error_m = lon_error * 111320 * np.cos(np.radians(true[:, 1]))
    lat_error_m = lat_error * 110540

    # 总误差（欧氏距离）
    total_error_m = np.sqrt(lon_error_m**2 + lat_error_m**2)

    print("\n" + "="*60)
    print("模型评估结果")
    print("="*60)
    print(f"平均经度误差: {np.mean(lon_error)*1e6:.2f} 微度 ({np.mean(lon_error_m):.2f} 米)")
    print(f"平均纬度误差: {np.mean(lat_error)*1e6:.2f} 微度 ({np.mean(lat_error_m):.2f} 米)")
    print(f"平均总误差: {np.mean(total_error_m):.2f} 米")
    print(f"最大误差: {np.max(total_error_m):.2f} 米")
    print(f"最小误差: {np.min(total_error_m):.2f} 米")
    print(f"RMS误差: {np.sqrt(np.mean(total_error_m**2)):.2f} 米")
    print(f"中位数误差: {np.median(total_error_m):.2f} 米")
    print("="*60)

    return pred, true, total_error_m

def plot_results(train_losses, val_losses, pred, true, errors):
    """绘制结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 训练和验证损失曲线
    axes[0, 0].plot(train_losses, label='Training Loss', linewidth=2)
    axes[0, 0].plot(val_losses, label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 预测vs真实经度
    axes[0, 1].scatter(true[:, 0], pred[:, 0], alpha=0.5, s=10, c='blue', label='Predictions')
    min_lon = min(true[:, 0].min(), pred[:, 0].min())
    max_lon = max(true[:, 0].max(), pred[:, 0].max())
    axes[0, 1].plot([min_lon, max_lon], [min_lon, max_lon], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 1].set_xlabel('True Longitude', fontsize=12)
    axes[0, 1].set_ylabel('Predicted Longitude', fontsize=12)
    axes[0, 1].set_title('Longitude Prediction', fontsize=14)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 预测vs真实纬度
    axes[1, 0].scatter(true[:, 1], pred[:, 1], alpha=0.5, s=10, c='green', label='Predictions')
    min_lat = min(true[:, 1].min(), pred[:, 1].min())
    max_lat = max(true[:, 1].max(), pred[:, 1].max())
    axes[1, 0].plot([min_lat, max_lat], [min_lat, max_lat], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 0].set_xlabel('True Latitude', fontsize=12)
    axes[1, 0].set_ylabel('Predicted Latitude', fontsize=12)
    axes[1, 0].set_title('Latitude Prediction', fontsize=14)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 误差分布
    axes[1, 1].hist(errors, bins=30, color='orange', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}m')
    axes[1, 1].axvline(np.median(errors), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.2f}m')
    axes[1, 1].set_xlabel('Error (meters)', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Prediction Error Distribution', fontsize=14)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lstm_training_results.png', dpi=300, bbox_inches='tight')
    print("\n结果图已保存为: lstm_training_results.png")
    plt.show()

def main():
    # 超参数配置
    SEQ_LENGTH = 10  # 序列长度（使用过去10个时间点预测当前）
    HIDDEN_SIZE = 64  # LSTM隐藏层大小
    NUM_LAYERS = 2  # LSTM层数
    BATCH_SIZE = 32  # 批次大小
    EPOCHS = 100  # 训练轮数
    LEARNING_RATE = 0.001  # 学习率
    DROPOUT = 0.2  # Dropout比率
    TEST_SIZE = 0.2  # 测试集比例

    print("GPS精度改善 - LSTM神经网络")
    print("="*60)

    # 1. 加载和预处理数据
    X_scaled, y_scaled, feature_scaler, target_scaler, df = load_and_preprocess_data('gps_train_dataset.csv')

    # 2. 创建时序序列
    print(f"\n创建时序序列（窗口大小={SEQ_LENGTH}）...")
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)
    print(f"序列数据形状: X={X_seq.shape}, y={y_seq.shape}")

    # 3. 划分训练集、验证集和测试集
    X_temp, X_test, y_temp, y_test = train_test_split(X_seq, y_seq, test_size=TEST_SIZE, random_state=42, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42, shuffle=False)

    print(f"\n数据集划分:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  验证集: {len(X_val)} 样本")
    print(f"  测试集: {len(X_test)} 样本")

    # 4. 创建DataLoader
    train_dataset = GPSDataset(X_train, y_train)
    val_dataset = GPSDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. 创建模型
    model = LSTMModel(
        input_size=3,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=2,
        dropout=DROPOUT
    )

    print(f"\n模型结构:")
    print(model)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 6. 训练模型
    model, train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE
    )

    # 7. 评估模型
    print("\n在测试集上评估模型...")
    pred, true, errors = evaluate_model(model, X_test, y_test, target_scaler, feature_scaler)

    # 8. 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'seq_length': SEQ_LENGTH,
        'model_config': {
            'input_size': 3,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'output_size': 2,
            'dropout': DROPOUT
        }
    }, 'lstm_gps_model.pth')
    print("\n模型已保存为: lstm_gps_model.pth")

    # 9. 绘制结果
    plot_results(train_losses, val_losses, pred, true, errors)

    # 10. 对比原始GPS误差
    print("\n对比原始GPS测量误差:")
    print("="*60)

    # 从测试集中获取对应的原始GPS数据
    test_indices = range(len(X_seq) - len(X_test), len(X_seq))
    original_gps = []
    for idx in test_indices:
        # 获取序列的最后一个时间步的原始GPS值
        gps_idx = idx + SEQ_LENGTH - 1
        if gps_idx < len(X_scaled):
            original_gps.append(X_scaled[gps_idx])

    if len(original_gps) == len(y_test):
        original_gps = np.array(original_gps)
        original_gps_unscaled = feature_scaler.inverse_transform(original_gps)

        # 原始GPS的经纬度
        original_lon = original_gps_unscaled[:, 0]
        original_lat = original_gps_unscaled[:, 1]

        # 计算原始误差
        orig_lon_error = np.abs(original_lon - true[:, 0]) * 111320 * np.cos(np.radians(true[:, 1]))
        orig_lat_error = np.abs(original_lat - true[:, 1]) * 110540
        orig_total_error = np.sqrt(orig_lon_error**2 + orig_lat_error**2)

        print(f"原始GPS平均误差: {np.mean(orig_total_error):.2f} 米")
        print(f"LSTM预测平均误差: {np.mean(errors):.2f} 米")
        print(f"精度提升: {(1 - np.mean(errors)/np.mean(orig_total_error))*100:.2f}%")
        print("="*60)

if __name__ == "__main__":
    main()