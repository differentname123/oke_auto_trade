import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from lightgbm import LGBMClassifier

from get_feature import get_latest_data


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

# ===================== 预测器类 =====================
class Predictor:
    def __init__(self, models_dir, seq_len, TARGET_COL):
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # 初始化参数
        self.seq_len = seq_len
        self.TARGET_COL = TARGET_COL
        self.key_name = f"{seq_len}_{TARGET_COL}"
        self.models_dir = models_dir
        self.reports_dir = "reports/BTC-USDT-SWAP_1m_20230124_20241218_features_tail"

        # 初始化模型和Scaler为None
        self.scaler = None
        self.lstm_model = None
        self.transformer_model = None
        self.lgbm_model = None
        self.meta_model = None

    def load_models_and_scaler(self, input_size):
        # 加载Scaler
        self.scaler = self.load_scaler()

        # 加载模型
        self.lstm_model = self.load_lstm_model(input_size)
        self.transformer_model = self.load_transformer_model(input_size)
        self.lgbm_model = self.load_lgbm_model()
        self.meta_model = self.load_meta_model()

    def load_scaler(self):
        scaler_path = os.path.join(self.models_dir, f"{self.key_name}_scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler文件未找到，请确保已在训练过程中保存了Scaler至 {scaler_path}")
        scaler = joblib.load(scaler_path)
        print(f"Scaler已从 {scaler_path} 加载")
        return scaler

    def load_lstm_model(self, input_size):
        lstm_model = LSTMModel(input_size=input_size, hidden_size=128, num_layers=2).to(self.device)
        lstm_model_path = os.path.join(self.models_dir, f"{self.key_name}_lstm_model.pth")
        if not os.path.exists(lstm_model_path):
            raise FileNotFoundError(f"LSTM模型文件未找到，请确保存在 {lstm_model_path}")
        lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=self.device))
        lstm_model.eval()
        print(f"LSTM模型已从 {lstm_model_path} 加载")
        return lstm_model

    def load_transformer_model(self, input_size):
        transformer_model = TransformerModel(input_size=input_size, num_layers=2, d_model=128, nhead=8, dim_feedforward=256).to(self.device)
        transformer_model_path = os.path.join(self.models_dir, f"{self.key_name}_transformer_model.pth")
        if not os.path.exists(transformer_model_path):
            raise FileNotFoundError(f"Transformer模型文件未找到，请确保存在 {transformer_model_path}")
        transformer_model.load_state_dict(torch.load(transformer_model_path, map_location=self.device))
        transformer_model.eval()
        print(f"Transformer模型已从 {transformer_model_path} 加载")
        return transformer_model

    def load_lgbm_model(self):
        lgbm_model_path = os.path.join(self.models_dir, f"{self.key_name}_lightgbm_model.pkl")
        if not os.path.exists(lgbm_model_path):
            raise FileNotFoundError(f"LightGBM模型文件未找到，请确保存在 {lgbm_model_path}")
        lgbm_model = joblib.load(lgbm_model_path)
        print(f"LightGBM模型已从 {lgbm_model_path} 加载")
        return lgbm_model

    def load_meta_model(self):
        meta_model_path = os.path.join(self.models_dir, f"{self.key_name}_meta_model.pkl")
        if not os.path.exists(meta_model_path):
            raise FileNotFoundError(f"Meta模型文件未找到，请确保存在 {meta_model_path}")
        meta_model = joblib.load(meta_model_path)
        print(f"Meta模型已从 {meta_model_path} 加载")
        return meta_model

    def load_and_preprocess_data(self, new_data_path):

        # 如果new_data_path已经是dataframe，直接读取
        if isinstance(new_data_path, pd.DataFrame):
            data = new_data_path
        else:
            if not os.path.exists(new_data_path):
                raise FileNotFoundError(f"新数据文件未找到，请确保存在 {new_data_path}")
            data = pd.read_csv(new_data_path)

        # 将timestamp设为索引（如果需要）
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.set_index('timestamp').sort_index()

        # 找到列名中包含 "close_down" 或 "close_up" 的列，除了TARGET_COL
        feature_cols = [col for col in data.columns if ('close_down' in col or 'close_up' in col) and col != self.TARGET_COL]

        # 删除feature_cols中的列
        data = data.drop(columns=feature_cols)

        # 处理无效值
        mask = (data.replace([np.inf, -np.inf], np.nan).isnull().any(axis=1)) | \
               ((data > np.finfo('float64').max).any(axis=1)) | \
               ((data < np.finfo('float64').min).any(axis=1))

        print(f"Removing {mask.sum()} rows with invalid values")
        data = data[~mask]

        # 特征与标签
        features = data.drop(columns=[self.TARGET_COL])
        labels = data[self.TARGET_COL]

        # 获取labels的分布比例
        label_counts = labels.value_counts(normalize=True)
        print("Labels distribution:", label_counts)

        # 对特征进行缩放处理（使用已加载的Scaler）
        features_scaled = self.scaler.transform(features)
        features_scaled = pd.DataFrame(features_scaled, index=features.index, columns=features.columns)

        return data, features_scaled, labels, label_counts

    def get_model_predictions(self, model, loader, model_type='pytorch'):
        model.eval()
        preds = []
        indices = []
        with torch.no_grad():
            for X_batch, y_batch, idx in loader:
                X_batch = X_batch.to(self.device)
                if model_type == 'pytorch':
                    outputs = model(X_batch)
                    prob = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds.append(prob)
                indices.extend(idx.numpy())
        return np.concatenate(preds), np.array(indices)

    def prepare_result_df(self, data, indices, labels_aligned, final_probs):
        if np.max(indices) >= len(data):
            raise IndexError("indices 中存在超出 data 索引范围的值")

        try:
            # 获取对应的时间戳
            timestamps = data.index[indices]
            # 将时间戳转换为字符串格式，确保在CSV中正确显示
            timestamps_str = timestamps.strftime('%Y-%m-%d %H:%M:%S')

            # 创建结果DataFrame
            results_df = pd.DataFrame({
                'timestamp': timestamps_str,
                'true_label': labels_aligned,
                'pred_final_prob': final_probs
            })
        except Exception as e:
            print(f"保存预测结果时出现异常：{e}")
            results_df = None

        return results_df

    def predict(self, new_data_path):
        # 如果Scaler未加载，先加载Scaler
        if self.scaler is None:
            self.scaler = self.load_scaler()

        # 加载并预处理数据
        data, features_scaled, labels, label_counts = self.load_and_preprocess_data(new_data_path)

        # 获取input_size
        input_size = features_scaled.shape[1]

        # 如果模型未加载，加载模型
        if self.lstm_model is None or self.transformer_model is None or self.lgbm_model is None or self.meta_model is None:
            self.load_models_and_scaler(input_size)

        # 创建数据集和数据加载器
        dataset = TimeSeriesDataset(features_scaled, labels, seq_len=self.seq_len, offset=0)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        # 获取模型预测
        print("Generating predictions with LSTM model...")
        preds_lstm, indices_lstm = self.get_model_predictions(self.lstm_model, loader, model_type='pytorch')

        print("Generating predictions with Transformer model...")
        preds_transformer, _ = self.get_model_predictions(self.transformer_model, loader, model_type='pytorch')

        print("Generating predictions with LightGBM model...")
        # LightGBM不需要序列数据，只需要特征
        X_lgbm = features_scaled.values[self.seq_len:]
        preds_lgbm = self.lgbm_model.predict_proba(X_lgbm)[:, 1]
        indices_lgbm = np.arange(self.seq_len, len(features_scaled))

        # 确保索引对齐
        if not (np.array_equal(indices_lstm, indices_lgbm)):
            raise ValueError("模型预测的索引不一致，无法进行Stacking")

        # 获取真实标签
        labels_aligned = labels.values[self.seq_len:]

        # Stacking特征
        stack_X = np.vstack([preds_lstm, preds_transformer, preds_lgbm]).T

        # 使用Meta模型进行最终预测
        final_probs = self.meta_model.predict_proba(stack_X)[:, 1]

        # 准备结果DataFrame
        result_df = self.prepare_result_df(data, indices_lstm, labels_aligned, final_probs)

        return result_df

# ===================== 使用示例 =====================
if __name__ == "__main__":
    new_data_path = "kline_data/df_features.csv"


    # 参数设置
    seq_len1 = 100
    TARGET_COL1 = "close_up_0.2_t9"
    models_dir = f"models/BTC-USDT-SWAP_1m_20230124_20241218_features_tail/{seq_len1}_{TARGET_COL1}"

    # 创建预测器实例
    predictor1 = Predictor(models_dir, seq_len1, TARGET_COL1)

    # 参数设置
    seq_len2 = 200
    TARGET_COL2 = "close_up_0.2_t9"
    models_dir = f"models/BTC-USDT-SWAP_1m_20230124_20241218_features_tail/{seq_len2}_{TARGET_COL2}"

    # 创建预测器实例
    predictor2 = Predictor(models_dir, seq_len2, TARGET_COL2)

    # 参数设置
    seq_len3 = 300
    TARGET_COL3 = "close_up_0.2_t9"
    models_dir = f"models/BTC-USDT-SWAP_1m_20230124_20241218_features_tail/{seq_len3}_{TARGET_COL3}"

    # 创建预测器实例
    predictor3 = Predictor(models_dir, seq_len3, TARGET_COL3)

    while True:
        feature_df = get_latest_data(max_candles=1000)
        # 进行预测
        result_df1 = predictor1.predict(new_data_path)
        result_df2 = predictor2.predict(new_data_path)
        result_df3 = predictor3.predict(new_data_path)

        result_df = pd.merge(result_df1, result_df2, on='timestamp', how='outer', suffixes=('_model1', '_model2'))
        result_df = pd.merge(result_df, result_df3, on='timestamp', how='outer', suffixes=('', '_model3'))

        # 如果需要处理列名后缀冲突（例如多次出现相同的列名），可以重命名列
        result_df = result_df.rename(columns=lambda x: x.replace('_model3', '_model3').strip('_'))
        # 输出结果
        print(result_df.tail())