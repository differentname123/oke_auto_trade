import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import re

# 定义与训练时相同的模型结构
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后时刻的输出
        out = self.fc(out)
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_layers=2, d_model=128, nhead=4, dim_feedforward=256, num_classes=2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=0.2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer_encoder(x)
        out = out.mean(dim=1)  # 取平均值作为输出
        out = self.fc(out)
        return out

# 定义PyTorch数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len=30, offset=0):
        self.X = torch.tensor(X, dtype=torch.float32) if isinstance(X, np.ndarray) else torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long) if isinstance(y, pd.Series) else torch.tensor(y, dtype=torch.long)
        self.seq_len = seq_len
        self.offset = offset

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        X_seq = self.X[idx: idx + self.seq_len]
        y_label = self.y[idx + self.seq_len]
        absolute_idx = self.offset + idx + self.seq_len
        return X_seq, y_label, absolute_idx

def generate_filename(params):
    """
    生成包含完整参数的文件名。
    格式: seq_len=100_lstm_hidden_size=128_lstm_lr=0.0005_lstm_num_layers=2_transformer_d_model=128_transformer_nhead=4_transformer_lr=0.0005_transformer_num_layers=2_lgbm_n_estimators=300_lgbm_learning_rate=0.01_meta_model_n_estimators=50
    """
    base_params = []
    base_params.append(f"seq_len={params['seq_len']}")
    base_params.append(f"lstm_hidden_size={params['lstm_hidden_size']}")
    base_params.append(f"lstm_lr={params['lstm_lr']}")
    base_params.append(f"lstm_num_layers={params['lstm_num_layers']}")
    base_params.append(f"transformer_d_model={params['transformer_d_model']}")
    base_params.append(f"transformer_nhead={params['transformer_nhead']}")
    base_params.append(f"transformer_lr={params['transformer_lr']}")
    base_params.append(f"transformer_num_layers={params['transformer_num_layers']}")
    base_params.append(f"lgbm_n_estimators={params['lgbm_n_estimators']}")
    base_params.append(f"lgbm_learning_rate={params['lgbm_learning_rate']}")
    base_params.append(f"meta_model_n_estimators={params['meta_model_n_estimators']}")
    params_str = "_".join(base_params)
    return params_str

def load_models(params, models_dir, data_name, target_col):
    """
    根据参数组合加载相应的模型和Scaler
    """
    # 生成文件名
    params_str = generate_filename(params)
    filename_lstm = f"lstm_{params_str}.pth"
    filename_transformer = f"transformer_{params_str}.pth"
    filename_lgbm = f"lightgbm_{params_str}.pkl"
    filename_meta = f"meta_model_{params_str}.pkl"
    filename_scaler = f"scaler_{params_str}.pkl"

    # 定位到相应的子文件夹
    model_subdir = os.path.join(models_dir, data_name, target_col)

    # 加载Scaler
    scaler_path = os.path.join(model_subdir, filename_scaler)
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler文件未找到: {scaler_path}")
    scaler = joblib.load(scaler_path)

    # 加载LSTM模型
    lstm_path = os.path.join(model_subdir, filename_lstm)
    if not os.path.exists(lstm_path):
        raise FileNotFoundError(f"LSTM模型文件未找到: {lstm_path}")
    lstm_model = LSTMModel(input_size=params['input_size'],
                           hidden_size=params['lstm_hidden_size'],
                           num_layers=params['lstm_num_layers']).to('cpu')
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=torch.device('cpu')))
    lstm_model.eval()

    # 加载Transformer模型
    transformer_path = os.path.join(model_subdir, filename_transformer)
    if not os.path.exists(transformer_path):
        raise FileNotFoundError(f"Transformer模型文件未找到: {transformer_path}")
    transformer_model = TransformerModel(
        input_size=params['input_size'],
        num_layers=params['transformer_num_layers'],
        d_model=params['transformer_d_model'],
        nhead=params['transformer_nhead']
    ).to('cpu')
    transformer_model.load_state_dict(torch.load(transformer_path, map_location=torch.device('cpu')))
    transformer_model.eval()

    # 加载LightGBM模型
    lgbm_path = os.path.join(model_subdir, filename_lgbm)
    if not os.path.exists(lgbm_path):
        raise FileNotFoundError(f"LightGBM模型文件未找到: {lgbm_path}")
    lgbm_model = joblib.load(lgbm_path)

    # 加载元模型
    meta_model_path = os.path.join(model_subdir, filename_meta)
    if not os.path.exists(meta_model_path):
        raise FileNotFoundError(f"Meta模型文件未找到: {meta_model_path}")
    meta_model = joblib.load(meta_model_path)

    return scaler, lstm_model, transformer_model, lgbm_model, meta_model

def preprocess_input(df, scaler, seq_len, target_col):
    """
    对输入的DataFrame进行预处理，与训练时保持一致
    """
    # 处理timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()

    # 找到列名中包含 "close_down" 或 "close_up" 的列，除了TARGET_COL
    feature_cols = [col for col in df.columns if ('close_down' in col or 'close_up' in col) and col != target_col]

    # 删除feature_cols中的列
    df = df.drop(columns=feature_cols, errors='ignore')

    # 处理缺失值和异常值
    mask = (df.replace([np.inf, -np.inf], np.nan).isnull().any(axis=1)) | \
           ((df > np.finfo('float64').max).any(axis=1)) | \
           ((df < np.finfo('float64').min).any(axis=1))
    df = df[~mask]

    # 特征缩放
    features = df.drop(columns=[target_col], errors='ignore')  # 如果有TARGET_COL则删除
    features_scaled = scaler.transform(features)
    features_scaled = pd.DataFrame(features_scaled, index=features.index, columns=features.columns)

    return features_scaled

def get_model_predictions(model, loader):
    model.eval()
    preds = []
    indices = []
    with torch.no_grad():
        for X_batch, _, idx in loader:
            outputs = model(X_batch)
            prob = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds.append(prob)
            indices.extend(idx.numpy())
    return np.concatenate(preds), np.array(indices)

def predict(df, model_filename, models_dir, data_file, target_col, input_size):
    """
    使用指定模型文件进行预测，并输出不同阈值下的报告
    """
    # 解析参数组合
    params_str_pattern = r"^(lstm|transformer|lightgbm|meta_model|scaler)_seq_len=(\d+)_lstm_hidden_size=(\d+)_lstm_lr=([0-9.e-]+)_lstm_num_layers=(\d+)_transformer_d_model=(\d+)_transformer_nhead=(\d+)_transformer_lr=([0-9.e-]+)_transformer_num_layers=(\d+)_lgbm_n_estimators=(\d+)_lgbm_learning_rate=([0-9.e-]+)_meta_model_n_estimators=(\d+)\.(pth|pkl|txt)$"
    match = re.match(params_str_pattern, model_filename)
    if not match:
        raise ValueError("无法解析模型文件名。请确保文件名格式为 'model_type_seq_len=<value>_lstm_hidden_size=<value>_... .ext'")

    # 提取参数
    (
        model_type,
        seq_len,
        lstm_hidden_size,
        lstm_lr,
        lstm_num_layers,
        transformer_d_model,
        transformer_nhead,
        transformer_lr,
        transformer_num_layers,
        lgbm_n_estimators,
        lgbm_learning_rate,
        meta_model_n_estimators,
        ext
    ) = match.groups()

    # 构建参数字典
    params = {
        'seq_len': int(seq_len),
        'lstm_hidden_size': int(lstm_hidden_size),
        'lstm_lr': float(lstm_lr),
        'lstm_num_layers': int(lstm_num_layers),
        'transformer_d_model': int(transformer_d_model),
        'transformer_nhead': int(transformer_nhead),
        'transformer_lr': float(transformer_lr),
        'transformer_num_layers': int(transformer_num_layers),
        'lgbm_n_estimators': int(lgbm_n_estimators),
        'lgbm_learning_rate': float(lgbm_learning_rate),
        'meta_model_n_estimators': int(meta_model_n_estimators),
        'input_size': input_size
    }

    # 生成文件名格式（确保与训练时一致）
    params_str = generate_filename(params)

    # 加载模型和Scaler
    scaler, lstm_model, transformer_model, lgbm_model, meta_model = load_models(params, models_dir, os.path.splitext(os.path.basename(data_file))[0], target_col)

    # 预处理输入数据
    features_scaled = preprocess_input(df, scaler, params['seq_len'], target_col)

    # 获取真实标签
    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不存在于输入数据中。请确保输入数据包含目标列。")
    true_labels = df[target_col].values[params['seq_len']:]
    if len(true_labels) == 0:
        raise ValueError("目标列的长度不足以进行预测。请检查 'seq_len' 参数和输入数据的长度。")

    # 创建数据集和加载器
    dataset = TimeSeriesDataset(features_scaled, df[target_col], seq_len=params['seq_len'], offset=0)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # 获取模型预测
    lstm_preds, _ = get_model_predictions(lstm_model, loader)
    transformer_preds, _ = get_model_predictions(transformer_model, loader)
    lgbm_preds = lgbm_model.predict_proba(features_scaled.values[params['seq_len']:])[:, 1]

    # 确保所有预测的长度一致
    min_length = min(len(lstm_preds), len(transformer_preds), len(lgbm_preds))
    stack_X = np.vstack([lstm_preds[:min_length], transformer_preds[:min_length], lgbm_preds[:min_length]]).T

    # 使用元模型进行预测，获取预测概率
    final_probs = meta_model.predict_proba(stack_X)[:, 1]

    # 获取对应的真实标签
    true_labels_aligned = true_labels[:min_length]

    # 定义阈值范围
    thresholds = np.arange(0.5, 1, 0.01)

    # 收集所有报告
    all_reports = []

    for threshold in thresholds:
        report_str = f"\nThreshold: {threshold:.2f}\n"
        print(report_str)

        # 生成预测类别
        final_preds_adjusted = np.where(final_probs >= threshold, 1, 0)

        # 计算分类报告
        report = classification_report(true_labels_aligned, final_preds_adjusted, digits=4)
        print(report)
        report_str += report + "\n"

        # 计算每个类别的准确率
        cm = confusion_matrix(true_labels_aligned, final_preds_adjusted)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        for idx, acc in enumerate(per_class_acc):
            acc_str = f"Accuracy for class {idx}: {acc:.4f}\n"
            print(acc_str)
            report_str += acc_str

        # 添加分隔符
        report_str += "="*50 + "\n\n"

        all_reports.append(report_str)

    print("预测完成。")

    # 保存报告
    report_output_dir = os.path.join("reports", os.path.splitext(os.path.basename(data_file))[0], target_col)
    os.makedirs(report_output_dir, exist_ok=True)
    report_output_path = os.path.join(report_output_dir, f"performance_report_{params_str}.txt")
    with open(report_output_path, "w") as f:
        for report in all_reports:
            f.write(report)
    print(f"性能报告已保存至 {report_output_path}")

    # 保存预测结果
    final_preds = (final_probs >= 0.5).astype(int)  # 默认阈值为0.5
    new_data_with_preds = df.copy()
    new_data_with_preds['predictions'] = np.nan
    new_data_with_preds.iloc[params['seq_len']:, -1] = final_preds  # 替换为实际的列索引或名称
    new_data_with_preds.to_csv("new_data_with_predictions.csv", index=False)
    print("预测结果已保存至 new_data_with_predictions.csv")

if __name__ == "__main__":
    # 定义必要的变量
    models_dir = "models"
    data_file = "BTC-USDT-SWAP_1m_20230124_20241218_features_tail.csv"
    data_name = os.path.splitext(os.path.basename(data_file))[0]
    target_col = "close_up_0.35_t60"

    # 读取待预测数据
    new_data_file = "models/test_data_with_labels.csv"  # 替换为您的数据文件路径
    new_data = pd.read_csv(new_data_file)

    # 指定要使用的模型文件名
    # 例如：'lightgbm_seq_len=100_lstm_hidden_size=128_lstm_lr=0.0005_lstm_num_layers=2_transformer_d_model=128_transformer_nhead=4_transformer_lr=0.0005_transformer_num_layers=2_lgbm_n_estimators=300_lgbm_learning_rate=0.01_meta_model_n_estimators=50.pkl'
    model_filename = "lightgbm_seq_len=100_lstm_hidden_size=128_lstm_lr=0.0005_lstm_num_layers=2_transformer_d_model=128_transformer_nhead=4_transformer_lr=0.0005_transformer_num_layers=2_lgbm_n_estimators=300_lgbm_learning_rate=0.01_meta_model_n_estimators=50.pkl"  # 替换为实际的模型文件名

    # 设置输入大小（应与训练时一致）
    # 例如，如果训练时的特征数量为 50，则设置 input_size = 50
    input_size = 94  # 替换为实际的特征数量

    # 进行预测
    try:
        final_probs, all_reports = predict(
            df=new_data,
            model_filename=model_filename,
            models_dir=models_dir,
            data_file=data_file,
            target_col=target_col,
            input_size=input_size
        )

        # 输出并保存报告
        report_output_dir = os.path.join("reports", data_name, target_col)
        os.makedirs(report_output_dir, exist_ok=True)
        report_output_path = os.path.join(report_output_dir, f"performance_report_{generate_filename({'seq_len':100, 'lstm_hidden_size':128, 'lstm_lr':0.0005, 'lstm_num_layers':2, 'transformer_d_model':128, 'transformer_nhead':4, 'transformer_lr':0.0005, 'transformer_num_layers':2, 'lgbm_n_estimators':300, 'lgbm_learning_rate':0.01, 'meta_model_n_estimators':50})}.txt")
        with open(report_output_path, "w") as f:
            for report in all_reports:
                f.write(report)
        print(f"性能报告已保存至 {report_output_path}")

        # 保存预测结果
        final_preds = (final_probs >= 0.5).astype(int)  # 默认阈值为0.5
        new_data_with_preds = new_data.copy()
        new_data_with_preds['predictions'] = np.nan
        new_data_with_preds.iloc[100:, -1] = final_preds  # 替换为实际的列索引或名称
        new_data_with_preds.to_csv("new_data_with_predictions.csv", index=False)
        print("预测结果已保存至 new_data_with_predictions.csv")

    except Exception as e:
        print(f"预测过程中发生错误: {e}")