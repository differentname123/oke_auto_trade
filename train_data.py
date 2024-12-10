import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.utils.class_weight import compute_class_weight

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# === Step 1: 加载数据 ===
file_path = "BTC-USDT-SWAP_1m_20230116_20241210.csv"
data = pd.read_csv(file_path)
data = data.sort_values(by="timestamp").reset_index(drop=True)
# data = data.tail(50000)

# === Step 2: 数据预处理 ===
features = ["open", "high", "low", "close", "volume"]


# 添加技术指标特征
def add_technical_indicators(df):
    # 移动平均线
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    # 价格变化率
    df['returns'] = df['close'].pct_change()
    # 波动率
    df['volatility'] = df['returns'].rolling(window=10).std()
    # RSI 相对强弱指数
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()
    rs = roll_up / roll_down
    df['rsi'] = 100.0 - (100.0 / (1.0 + rs))
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    return df


data = add_technical_indicators(data)

# 更新特征列表
features += ['ma5', 'ma10', 'ma20', 'returns', 'volatility', 'rsi', 'macd']
target = "close_up_0.03_t5"

# 打印目标变量比例
print("目标变量分布：")
print(data[target].value_counts(normalize=True))

# 处理缺失值
data = data.dropna()

# 标准化特征
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# === Step 3: 创建时间序列数据集 ===
sequence_length = 60


class TimeSeriesDataset(Dataset):
    def __init__(self, data, features, target, sequence_length):
        self.X, self.y = [], []
        data_features = data[features].values
        data_target = data[target].values
        for i in range(len(data) - sequence_length):
            self.X.append(data_features[i:i + sequence_length])
            self.y.append(data_target[i + sequence_length])
        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


# 划分训练集、验证集和测试集
train_val_data, test_data = train_test_split(data, shuffle=False, test_size=0.2)
train_data, val_data = train_test_split(train_val_data, shuffle=False, test_size=0.1)  # 验证集占训练集的10%

# 创建数据集
train_dataset = TimeSeriesDataset(train_data.reset_index(drop=True), features, target, sequence_length)
val_dataset = TimeSeriesDataset(val_data.reset_index(drop=True), features, target, sequence_length)
test_dataset = TimeSeriesDataset(test_data.reset_index(drop=True), features, target, sequence_length)

print(f"训练集样本数：{len(train_dataset)}")
print(f"验证集样本数：{len(val_dataset)}")
print(f"测试集样本数：{len(test_dataset)}")

# 创建数据加载器，不打乱数据
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# === Step 4: 定义优化后的 Transformer 模型 ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, feature_size, embed_dim, num_classes, num_heads, num_encoder_layers, dropout):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(feature_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(src.size(-1))
        src = self.pos_encoder(src)
        transformer_output = self.transformer_encoder(src)
        output = transformer_output[:, -1, :]  # 取最后一个时间步的输出
        output = self.dropout(output)
        output = self.fc(output)
        return output


# === Step 5: 定义损失函数和优化器 ===
model = TransformerModel(feature_size=len(features), embed_dim=64, num_classes=2, num_heads=4, num_encoder_layers=2,
                         dropout=0.2)

# 使用 CrossEntropyLoss 并正确设置类别权重
train_labels = train_dataset.y
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32)
print(f"类别权重：{class_weights}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# === Step 6: 训练模型 ===
num_epochs = 50
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)

    train_loss = train_loss / len(train_dataset)

    # 验证集评估
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs = model(X_val)
            loss = criterion(outputs, y_val)
            val_loss += loss.item() * X_val.size(0)
    val_loss = val_loss / len(val_dataset)

    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    scheduler.step(val_loss)

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# === Step 7: 评估模型 ===
model.eval()
y_pred_probs, y_true = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # 正类概率
        y_pred_probs.extend(probs.cpu().numpy())
        y_true.extend(y_batch.numpy())

y_pred_probs = np.array(y_pred_probs)
y_true = np.array(y_true)

# 自定义阈值列表
thresholds = [x * 0.02 for x in range(25, 50)]
print("自定义阈值列表：", thresholds)
for threshold in thresholds:
    y_pred = (y_pred_probs >= threshold).astype(int)
    print(f"\n阈值：{threshold}")
    print(classification_report(y_true, y_pred, digits=4))
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("混淆矩阵：")
    print(conf_matrix)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_true, y_pred_probs)
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

# 绘制精确率-召回率曲线
import matplotlib.pyplot as plt

precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred_probs)
plt.figure()
plt.plot(recall, precision, label='PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.show()