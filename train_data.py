import os
import pandas as pd
import numpy as np

# 导入必要的库
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
import joblib  # 用于保存LightGBM模型

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===================== 创建保存模型和报告的文件夹 =====================
models_dir = "models"
reports_dir = "reports"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

# ===================== 数据读入与预处理 =====================
# 假设数据文件名为 "BTC-USDT-SWAP_1m_20240627_20241212_features.csv"
data = pd.read_csv("BTC-USDT-SWAP_1m_20240627_20241212_features.csv")

# 假设数据中有:
# timestamp: 时间戳
# close_up_0.02_t5: 目标变量(0/1)
# 其他列为特征(价格与技术指标)
TARGET_COL = "close_up_0.09_t4"

# 将timestamp设为索引（如果需要）
if 'timestamp' in data.columns:
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp').sort_index()

# 找到列名中包含 "close_down" 或 "close_up" 的列，除了TARGET_COL
feature_cols = [col for col in data.columns if ('close_down' in col or 'close_up' in col) and col != TARGET_COL]

# 删除feature_cols中的列
data = data.drop(columns=feature_cols)

mask = (data.replace([np.inf, -np.inf], np.nan).isnull().any(axis=1)) | \
       ((data > np.finfo('float64').max).any(axis=1)) | \
       ((data < np.finfo('float64').min).any(axis=1))

print(f"Removing {mask.sum()} rows with invalid values")
# 删除这些行
data = data[~mask]

# 特征与标签
features = data.drop(columns=[TARGET_COL])
labels = data[TARGET_COL]

# 数据集划分（按比例切分，保证时间顺序，避免数据泄露）
train_size = 0.8
val_size = 0.1
test_size = 0.1

# 计算分割点索引
n_total = len(features)
train_end = int(n_total * train_size)
val_end = int(n_total * (train_size + val_size))

# 划分数据集
train_data = features.iloc[:train_end]
train_labels = labels.iloc[:train_end]

val_data = features.iloc[train_end:val_end]
val_labels = labels.iloc[train_end:val_end]

test_data = features.iloc[val_end:]
test_labels = labels.iloc[val_end:]


# ===================== 保存原始数据到文件 =====================

# 保存训练集数据
train_data_with_labels = pd.concat([train_data, train_labels], axis=1)
train_data_with_labels.to_csv(os.path.join(models_dir, "train_data_with_labels.csv"), index=False)

# 保存验证集数据
val_data_with_labels = pd.concat([val_data, val_labels], axis=1)
val_data_with_labels.to_csv(os.path.join(models_dir, "val_data_with_labels.csv"), index=False)

# 保存测试集数据
test_data_with_labels = pd.concat([test_data, test_labels], axis=1)
test_data_with_labels.to_csv(os.path.join(models_dir, "test_data_with_labels.csv"), index=False)

print("原始数据集已保存为 CSV 文件")


# 对特征进行缩放处理（仅使用训练集的统计量）
scaler = StandardScaler()
scaler.fit(train_data)

train_data_scaled = scaler.transform(train_data)
val_data_scaled = scaler.transform(val_data)
test_data_scaled = scaler.transform(test_data)

# 保存 StandardScaler
scaler_path = os.path.join(models_dir, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"Scaler 已保存至 {scaler_path}")


# 转换回DataFrame（可选）
train_data_scaled = pd.DataFrame(train_data_scaled, index=train_data.index, columns=train_data.columns)
val_data_scaled = pd.DataFrame(val_data_scaled, index=val_data.index, columns=train_data.columns)
test_data_scaled = pd.DataFrame(test_data_scaled, index=test_data.index, columns=train_data.columns)

# ===================== 构建PyTorch数据集 =====================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len=30, offset=0):
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

seq_len = 200

# 创建数据集，传入正确的 offset
train_dataset = TimeSeriesDataset(train_data_scaled, train_labels, seq_len=seq_len, offset=0)
val_dataset = TimeSeriesDataset(val_data_scaled, val_labels, seq_len=seq_len, offset=train_end)
test_dataset = TimeSeriesDataset(test_data_scaled, test_labels, seq_len=seq_len, offset=val_end)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ===================== 定义LSTM模型 =====================
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

# ===================== 定义Transformer模型 =====================
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

# ===================== 训练函数 =====================
def train_model(model, train_loader, val_loader, lr=1e-3, epochs=50, patience=5, class_weights=None):
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # 验证
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for X_batch, y_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                val_preds.append(preds.cpu().numpy())
                val_true.append(y_batch.cpu().numpy())
        val_preds = np.concatenate(val_preds)
        val_true = np.concatenate(val_true)
        acc = accuracy_score(val_true, val_preds)
        scheduler.step(acc)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Accuracy: {acc:.4f}")

        if acc > best_val_acc:
            best_val_acc = acc
            patience_counter = 0
            # 保存最优模型参数
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    if best_model_state is not None:
        # 加载最优模型参数
        model.load_state_dict(best_model_state)
    return best_val_acc

# 获取输入大小
input_size = train_data_scaled.shape[1]

# ===================== 处理类别不平衡 =====================
# 计算训练集中的类别权重
train_labels_np = train_labels.values[seq_len:]
class_counts = np.bincount(train_labels_np)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

print("Class weights:", class_weights)

# 计算LightGBM的scale_pos_weight
n_positive = np.sum(train_labels_np == 1)
n_negative = np.sum(train_labels_np == 0)
scale_pos_weight = n_negative / n_positive
print("Scale pos weight for LightGBM:", scale_pos_weight)

# ===================== 训练LSTM模型 =====================
lstm_model = LSTMModel(input_size=input_size, hidden_size=128, num_layers=2).to(device)
print("Training LSTM Model...")
val_acc_lstm = train_model(lstm_model, train_loader, val_loader, epochs=50, patience=10, lr=1e-3, class_weights=class_weights_tensor)

# 保存LSTM模型
lstm_model_path = os.path.join(models_dir, "lstm_model.pth")
torch.save(lstm_model.state_dict(), lstm_model_path)
print(f"LSTM模型已保存至 {lstm_model_path}")

# ===================== 训练Transformer模型 =====================
transformer_model = TransformerModel(input_size=input_size, num_layers=2, d_model=128, nhead=8, dim_feedforward=256).to(device)
print("Training Transformer Model...")
val_acc_transformer = train_model(transformer_model, train_loader, val_loader, epochs=50, patience=10, lr=1e-3, class_weights=class_weights_tensor)

# 保存Transformer模型
transformer_model_path = os.path.join(models_dir, "transformer_model.pth")
torch.save(transformer_model.state_dict(), transformer_model_path)
print(f"Transformer模型已保存至 {transformer_model_path}")

# ===================== 训练LightGBM模型 =====================
# 为LightGBM模型准备数据，无需序列化
X_train_np = train_data_scaled.values[seq_len:]  # 调整训练数据，以匹配时间序列模型的标签对齐
y_train_np = train_labels_np
X_val_np = val_data_scaled.values[seq_len:]  # 同样调整验证数据
y_val_np = val_labels.values[seq_len:]

lgbm = LGBMClassifier(n_estimators=500, scale_pos_weight=scale_pos_weight)

lgbm.fit(
    X_train_np,
    y_train_np,
    eval_set=[(X_val_np, y_val_np)],
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=1)
    ]
)

# 保存LightGBM模型
lgbm_model_path = os.path.join(models_dir, "lightgbm_model.pkl")
joblib.dump(lgbm, lgbm_model_path)
print(f"LightGBM模型已保存至 {lgbm_model_path}")

# ===================== 提取各模型预测用于Stacking =====================
def get_model_predictions(model, loader):
    model.eval()
    preds = []
    indices = []
    with torch.no_grad():
        for X_batch, _, idx in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            prob = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds.append(prob)
            indices.extend(idx.numpy())
    return np.concatenate(preds), np.array(indices)

# 获取验证集预测结果及对应绝对索引
val_preds_lstm, val_indices = get_model_predictions(lstm_model, val_loader)
val_preds_transformer, _ = get_model_predictions(transformer_model, val_loader)

# 获取LightGBM模型预测概率
val_preds_lgbm = lgbm.predict_proba(val_data_scaled.values[seq_len:])[:, 1]

# 对齐索引
val_indices_adjusted = val_indices - (train_end + seq_len)

# 获取对应的真实标签
val_labels_aligned = val_labels.values[val_indices - train_end]

# Stacking特征
stack_X_val = np.vstack([val_preds_lstm, val_preds_transformer, val_preds_lgbm[val_indices_adjusted]]).T
stack_y_val = val_labels_aligned

# 训练元模型（使用LightGBM作为元模型）
meta_model = LGBMClassifier(n_estimators=100)
meta_model.fit(stack_X_val, stack_y_val)

# 保存元模型
meta_model_path = os.path.join(models_dir, "meta_model.pkl")
joblib.dump(meta_model, meta_model_path)
print(f"元模型已保存至 {meta_model_path}")

# ===================== 测试集评估 =====================
# 获取测试集预测结果及对应绝对索引
test_preds_lstm, test_indices = get_model_predictions(lstm_model, test_loader)
test_preds_transformer, _ = get_model_predictions(transformer_model, test_loader)
test_preds_lgbm = lgbm.predict_proba(test_data_scaled.values[seq_len:])[:, 1]

# 对齐索引
test_indices_adjusted = test_indices - (val_end + seq_len)

# 获取对应的真实标签
test_labels_aligned = test_labels.values[test_indices - val_end]

# Stacking特征
stack_X_test = np.vstack([test_preds_lstm, test_preds_transformer, test_preds_lgbm[test_indices_adjusted]]).T
stack_y_test = test_labels_aligned

# 使用元模型进行预测，获取预测概率
final_probs = meta_model.predict_proba(stack_X_test)[:, 1]

# 定义阈值范围
thresholds = np.arange(0.5, 1, 0.01)

from sklearn.metrics import classification_report, confusion_matrix

# 收集所有报告
all_reports = []

for threshold in thresholds:
    report_str = f"\nThreshold: {threshold:.2f}\n"
    print(report_str)

    # 生成预测类别
    final_preds_adjusted = np.where(final_probs >= threshold, 1, 0)

    # 计算分类报告
    report = classification_report(stack_y_test, final_preds_adjusted, digits=4)
    print(report)
    report_str += report + "\n"

    # 计算每个类别的准确率
    cm = confusion_matrix(stack_y_test, final_preds_adjusted)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    for idx, acc in enumerate(per_class_acc):
        acc_str = f"Accuracy for class {idx}: {acc:.4f}\n"
        print(acc_str)
        report_str += acc_str

    all_reports.append(report_str)

print("Class weights:", class_weights)

# 将所有报告写入文件
report_file_path = os.path.join(reports_dir, "performance_report.txt")
with open(report_file_path, "w") as f:
    f.write("性能报告\n")
    f.write("="*50 + "\n\n")
    for report in all_reports:
        f.write(report)
        f.write("="*50 + "\n\n")
print(f"性能报告已保存至 {report_file_path}")
