import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from lightgbm import LGBMClassifier

# ===================== 设置设备和目录 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

models_dir = "models"
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

# ===================== 定义模型类 =====================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # 取最后时刻的输出
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_layers=2, d_model=128, nhead=8, dim_feedforward=256, num_classes=2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=0.2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.embedding(x)
        out = self.transformer_encoder(x)
        # 取平均值作为输出
        out = out.mean(dim=1)  # (batch, d_model)
        out = self.fc(out)
        return out

# ===================== 定义数据集类 =====================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len=200, offset=0):
        # X, y为DataFrame或ndarray
        self.X = torch.tensor(X.values, dtype=torch.float32) if hasattr(X, 'values') else torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long) if hasattr(y, 'values') else torch.tensor(y, dtype=torch.long)
        self.seq_len = seq_len
        self.offset = offset  # 当前数据集在整个数据集中的起始索引

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        X_seq = self.X[idx: idx + self.seq_len]
        y_label = self.y[idx + self.seq_len]
        absolute_idx = self.offset + idx + self.seq_len  # 计算在整个数据集中的绝对索引
        return X_seq, y_label, absolute_idx

# ===================== 加载预处理器 =====================
scaler_path = os.path.join(models_dir, "scaler.pkl")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler文件未找到，请确保已在训练过程中保存了Scaler至 {scaler_path}")
scaler = joblib.load(scaler_path)
print(f"Scaler已从 {scaler_path} 加载")

# ===================== 加载新数据并预处理 =====================
# 假设新数据文件名为 "new_data.csv"
new_data_path = "BTC-USDT-SWAP_1m_20241212_20241214_features.csv"
if not os.path.exists(new_data_path):
    raise FileNotFoundError(f"新数据文件未找到，请确保存在 {new_data_path}")

data = pd.read_csv(new_data_path)

# 假设数据中有:
# timestamp: 时间戳
# close_down_0.08_t5: 目标变量(0/1)
# 其他列为特征(价格与技术指标)
TARGET_COL = "close_down_0.08_t5"

# 将timestamp设为索引（如果需要）
if 'timestamp' in data.columns:
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp').sort_index()

# 找到列名中包含 "close_down" 或 "close_up" 的列，除了TARGET_COL
feature_cols = [col for col in data.columns if ('close_down' in col or 'close_up' in col) and col != TARGET_COL]

# 删除feature_cols中的列
data = data.drop(columns=feature_cols)

# 处理无效值
mask = (data.replace([np.inf, -np.inf], np.nan).isnull().any(axis=1)) | \
       ((data > np.finfo('float64').max).any(axis=1)) | \
       ((data < np.finfo('float64').min).any(axis=1))

print(f"Removing {mask.sum()} rows with invalid values")
# 删除这些行
data = data[~mask]

# 特征与标签
features = data.drop(columns=[TARGET_COL])
labels = data[TARGET_COL]

# 对特征进行缩放处理（使用已加载的Scaler）
features_scaled = scaler.transform(features)

# 转换回DataFrame（可选）
features_scaled = pd.DataFrame(features_scaled, index=features.index, columns=features.columns)

# ===================== 创建数据集和数据加载器 =====================
seq_len = 200  # 与训练时相同

dataset = TimeSeriesDataset(features_scaled, labels, seq_len=seq_len, offset=0)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# ===================== 加载已保存的模型 =====================
input_size = features_scaled.shape[1]

# 加载LSTM模型
lstm_model = LSTMModel(input_size=input_size, hidden_size=128, num_layers=2).to(device)
lstm_model_path = os.path.join(models_dir, "lstm_model.pth")
if not os.path.exists(lstm_model_path):
    raise FileNotFoundError(f"LSTM模型文件未找到，请确保存在 {lstm_model_path}")
lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=device))
lstm_model.eval()
print(f"LSTM模型已从 {lstm_model_path} 加载")

# 加载Transformer模型
transformer_model = TransformerModel(input_size=input_size, num_layers=2, d_model=128, nhead=8, dim_feedforward=256).to(device)
transformer_model_path = os.path.join(models_dir, "transformer_model.pth")
if not os.path.exists(transformer_model_path):
    raise FileNotFoundError(f"Transformer模型文件未找到，请确保存在 {transformer_model_path}")
transformer_model.load_state_dict(torch.load(transformer_model_path, map_location=device))
transformer_model.eval()
print(f"Transformer模型已从 {transformer_model_path} 加载")

# 加载LightGBM模型
lgbm_model_path = os.path.join(models_dir, "lightgbm_model.pkl")
if not os.path.exists(lgbm_model_path):
    raise FileNotFoundError(f"LightGBM模型文件未找到，请确保存在 {lgbm_model_path}")
lgbm_model = joblib.load(lgbm_model_path)
print(f"LightGBM模型已从 {lgbm_model_path} 加载")

# 加载Meta模型
meta_model_path = os.path.join(models_dir, "meta_model.pkl")
if not os.path.exists(meta_model_path):
    raise FileNotFoundError(f"Meta模型文件未找到，请确保存在 {meta_model_path}")
meta_model = joblib.load(meta_model_path)
print(f"Meta模型已从 {meta_model_path} 加载")

# ===================== 获取模型预测 =====================
def get_model_predictions(model, loader, model_type='pytorch'):
    model.eval()
    preds = []
    indices = []
    with torch.no_grad():
        for X_batch, y_batch, idx in loader:
            X_batch = X_batch.to(device)
            if model_type == 'pytorch':
                outputs = model(X_batch)
                prob = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            elif model_type == 'lightgbm':
                prob = model.predict_proba(X_batch.cpu().numpy())[:, 1]
            preds.append(prob)
            indices.extend(idx.numpy())
    return np.concatenate(preds), np.array(indices)

# 获取LSTM预测
print("Generating predictions with LSTM model...")
preds_lstm, indices_lstm = get_model_predictions(lstm_model, loader, model_type='pytorch')

# 获取Transformer预测
print("Generating predictions with Transformer model...")
preds_transformer, indices_transformer = get_model_predictions(transformer_model, loader, model_type='pytorch')

# 获取LightGBM预测
print("Generating predictions with LightGBM model...")
# LightGBM不需要序列数据，只需要特征
# 因此，直接使用features_scaled[seq_len:]
X_lgbm = features_scaled.values[seq_len:]
preds_lgbm = lgbm_model.predict_proba(X_lgbm)[:, 1]
# 对应的索引
indices_lgbm = np.arange(seq_len, len(features_scaled))

# 确保索引对齐
if not (np.array_equal(indices_lstm, indices_transformer) and np.array_equal(indices_lstm, indices_lgbm)):
    raise ValueError("模型预测的索引不一致，无法进行Stacking")

# 获取真实标签
labels_aligned = labels.values[seq_len:]

# ===================== Stacking特征 =====================
# 确保所有预测数组长度一致
if not (len(preds_lstm) == len(preds_transformer) == len(preds_lgbm)):
    raise ValueError("预测结果长度不一致，无法进行Stacking")

stack_X = np.vstack([preds_lstm, preds_transformer, preds_lgbm]).T
stack_y = labels_aligned

# ===================== 使用Meta模型进行最终预测 =====================
final_probs = meta_model.predict_proba(stack_X)[:, 1]

# ===================== 生成性能报告 =====================
# 定义阈值范围
thresholds = np.arange(0.5, 1, 0.01)

all_reports = []

for threshold in thresholds:
    report_str = f"\nThreshold: {threshold:.2f}\n"
    print(report_str)

    # 生成预测类别
    final_preds = np.where(final_probs >= threshold, 1, 0)

    # 计算分类报告
    report = classification_report(stack_y, final_preds, digits=4)
    print(report)
    report_str += report + "\n"

    # 计算每个类别的准确率
    cm = confusion_matrix(stack_y, final_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    for idx, acc in enumerate(per_class_acc):
        acc_str = f"Accuracy for class {idx}: {acc:.4f}\n"
        print(acc_str)
        report_str += acc_str

    all_reports.append(report_str)

base_name = os.path.basename(new_data_path)
# ===================== 保存性能报告 =====================
report_file_path = os.path.join(reports_dir, f"performance_report_{base_name}_{TARGET_COL}.txt")
with open(report_file_path, "w") as f:
    f.write("性能报告\n")
    f.write("="*50 + "\n\n")
    for report in all_reports:
        f.write(report)
        f.write("="*50 + "\n\n")
print(f"性能报告已保存至 {report_file_path}")
