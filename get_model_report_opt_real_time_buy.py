import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from lightgbm import LGBMClassifier

from common_utils import get_config
from get_feature import get_latest_data
import warnings

# 屏蔽 PyTorch的 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")


import os
import okx.Trade as Trade
import okx.MarketData as Market
import okx.Account as Account
from common_utils import get_config
import time
# 设置代理（如果需要）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

flag = "0"  # 实盘: 0, 模拟盘: 1

if flag == "1":
    # API 初始化
    apikey = get_config('api_key')
    secretkey = get_config('secret_key')
    passphrase = get_config('passphrase')
else:
    # API 初始化
    apikey = get_config('true_api_key')
    secretkey = get_config('true_secret_key')
    passphrase = get_config('true_passphrase')

# 初始化 OKX API 模块
tradeAPI = Trade.TradeAPI(apikey, secretkey, passphrase, False, flag)
marketAPI = Market.MarketAPI(apikey, secretkey, passphrase, False, flag)
accountAPI = Account.AccountAPI(apikey, secretkey, passphrase, False, flag)

# 全局配置
INST_ID = "BTC-USDT-SWAP"  # 交易对
ORDER_SIZE = 0.1            # 每次固定下单量


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
        # print("Using device:", self.device)

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
        # print(f"Scaler已从 {scaler_path} 加载")
        return scaler

    def load_lstm_model(self, input_size):
        lstm_model = LSTMModel(input_size=input_size, hidden_size=128, num_layers=2).to(self.device)
        lstm_model_path = os.path.join(self.models_dir, f"{self.key_name}_lstm_model.pth")
        if not os.path.exists(lstm_model_path):
            raise FileNotFoundError(f"LSTM模型文件未找到，请确保存在 {lstm_model_path}")
        lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=self.device))
        lstm_model.eval()
        # print(f"LSTM模型已从 {lstm_model_path} 加载")
        return lstm_model

    def load_transformer_model(self, input_size):
        transformer_model = TransformerModel(input_size=input_size, num_layers=2, d_model=128, nhead=8, dim_feedforward=256).to(self.device)
        transformer_model_path = os.path.join(self.models_dir, f"{self.key_name}_transformer_model.pth")
        if not os.path.exists(transformer_model_path):
            raise FileNotFoundError(f"Transformer模型文件未找到，请确保存在 {transformer_model_path}")
        transformer_model.load_state_dict(torch.load(transformer_model_path, map_location=self.device))
        transformer_model.eval()
        # print(f"Transformer模型已从 {transformer_model_path} 加载")
        return transformer_model

    def load_lgbm_model(self):
        lgbm_model_path = os.path.join(self.models_dir, f"{self.key_name}_lightgbm_model.pkl")
        if not os.path.exists(lgbm_model_path):
            raise FileNotFoundError(f"LightGBM模型文件未找到，请确保存在 {lgbm_model_path}")
        lgbm_model = joblib.load(lgbm_model_path)
        # print(f"LightGBM模型已从 {lgbm_model_path} 加载")
        return lgbm_model

    def load_meta_model(self):
        meta_model_path = os.path.join(self.models_dir, f"{self.key_name}_meta_model.pkl")
        if not os.path.exists(meta_model_path):
            raise FileNotFoundError(f"Meta模型文件未找到，请确保存在 {meta_model_path}")
        meta_model = joblib.load(meta_model_path)
        # print(f"Meta模型已从 {meta_model_path} 加载")
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

        # print(f"Removing {mask.sum()} rows with invalid values")
        data = data[~mask]

        # 特征与标签
        features = data.drop(columns=[self.TARGET_COL])
        labels = data[self.TARGET_COL]

        # 获取labels的分布比例
        label_counts = labels.value_counts(normalize=True)
        # print("Labels distribution:", label_counts)

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
                f'true_label_{self.key_name}': labels_aligned,
                f'pred_final_prob_{self.key_name}': final_probs
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
        # print("Generating predictions with LSTM model...")
        preds_lstm, indices_lstm = self.get_model_predictions(self.lstm_model, loader, model_type='pytorch')

        # print("Generating predictions with Transformer model...")
        preds_transformer, _ = self.get_model_predictions(self.transformer_model, loader, model_type='pytorch')

        # print("Generating predictions with LightGBM model...")
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

def place_order(inst_id, side, order_type, size, price=None, tp_price=None):
    """下单函数"""
    try:
        pos_side = "long" if side == "buy" else "short"

        # 构建下单参数
        order_params = {
            "instId": inst_id,
            "tdMode": "cross",  # 全仓模式
            "side": side,       # 买入或卖出
            "ordType": order_type,  # 订单类型：limit 或 market
            "sz": str(size),    # 下单量
            "posSide": pos_side  # 仓位方向：多头或空头
        }

        # 如果是限价单，添加价格参数
        if order_type == "limit" and price:
            order_params["px"] = str(price)

        # 如果需要止盈参数，添加止盈触发价格和订单价格
        if tp_price:
            order_params["tpTriggerPx"] = str(tp_price)
            order_params["tpOrdPx"] = str(tp_price)

        # 调用下单接口
        order = tradeAPI.place_order(**order_params)
        # 将 order 增量写入到文件
        with open("order_history.txt", "a") as f:
            f.write(str(order) + "\n")

        print(f"{side.upper()} 订单成功：", order)
        return order
    except Exception as e:
        print(f"{side.upper()} 订单失败，错误信息：", e)
        return None

def safe_api_call(api_function, *args, **kwargs):
    """安全调用 API，捕获异常"""
    try:
        return api_function(*args, **kwargs)
    except Exception as e:
        print(f"调用 {api_function.__name__} 失败：{e}")
        return None


def get_open_orders():
    """获取未完成的订单"""
    return safe_api_call(tradeAPI.get_order_list, instType="SWAP")

def cancel_order(inst_id, order_id):
    """撤销订单"""
    return safe_api_call(tradeAPI.cancel_order, instId=inst_id, ordId=order_id)

def release_funds(inst_id, latest_price, release_len):
    """根据当前价格和订单价格差距，取消指定数量的订单。"""
    open_orders = get_open_orders()

    sorted_orders = sorted(
        open_orders['data'],
        key=lambda order: abs(latest_price - float(order['px'])),
        reverse=True
    )

    for order in sorted_orders[:release_len]:
        order_id = order['ordId']
        order_price = float(order['px'])
        price_diff = abs(latest_price - order_price)
        last_updated = float(order['uTime']) / 1000  # 假设时间戳以毫秒为单位

        # 检查价格偏移和时间间隔
        if (time.time() - last_updated) <= 60:
            print(f"保留订单 {order_id}，订单价格：{order['px']}，最新价格：{latest_price}，差距：{price_diff}，时间间隔在1分钟内。")
            continue

        print(
            f"取消订单 {order_id}，订单价格：{order['px']}，最新价格：{latest_price}，差距：{price_diff}"
        )
        cancel_order(inst_id, order_id)


def compute_intersection_proportion(result_df, sort_columns, percentiles, target_column):
    """
    计算每个排序字段的前 `percentile` 部分的交集，并计算目标列 `target_column` 为 1 的占比。

    :param result_df: 原始 DataFrame
    :param sort_columns: 排序字段列表
    :param percentiles: 分段比例列表 (e.g., [0.1, 0.2, 0.3, 0.4, 0.5])
    :param target_column: 目标列名称 (e.g., '400_close_up_0.25_t9')
    :return: 每个分段的占比字典
    """
    proportions = {}
    result_list = []

    # 对每个百分比进行处理
    for percentile in percentiles:
        # 初始化一个布尔索引，用于存储所有排序字段的交集
        intersection_mask = pd.Series([True] * len(result_df), index=result_df.index)

        # 按照每个排序字段排序并取前 percentile 部分
        for col in sort_columns:
            # 排序并取前 percentile 部分的布尔索引
            sorted_df = result_df.sort_values(by=col, ascending=False)
            threshold_idx = int(len(sorted_df) * percentile)
            current_mask = sorted_df.index[:threshold_idx]

            # 对交集进行更新，取交集
            intersection_mask &= result_df.index.isin(current_mask)

        # 通过交集布尔索引，获取交集数据
        subset_df = result_df[intersection_mask]

        # 计算 '400_close_up_0.25_t9' 为 1 的占比
        proportion = (subset_df[target_column] == 1).mean()

        # 存储结果
        proportions[percentile] = proportion
        result_list.append({
            'percentile': percentile,
            'proportion': proportion,
            'count': len(subset_df)
        })

    return result_list


def analyze_data(result_df):
    sort_columns = ['pred_up_sum']  # 排序字段列表
    percentiles = [x / 200 for x in range(1, 30)]  # 分段比例列表
    percentiles.append(0.5)
    percentiles.append(1)
    target_column = '400_close_up_0.25_t9'  # 目标列

    proportions = compute_intersection_proportion(result_df, sort_columns, percentiles, target_column)

    # 输出每个分段的占比
    for percentile, proportion in proportions.items():
        print(f"前 {percentile * 100}% 数据中 '{target_column}' 为 1 的占比: {proportion:.4f}")
    # 输出每个分段的占比
    for percentile, proportion in proportions.items():
        print(f"前 {percentile * 100}% 数据中 '400_close_up_0.25_t9' 为 1 的占比: {proportion:.4f}")


def merge_results(results):
    # 合并多个预测结果并重命名列，防止列名冲突
    merged_df = results[0]
    for result in results[1:]:
        merged_df = pd.merge(merged_df, result, on='timestamp', how='outer')
    return merged_df

# ===================== 使用示例 =====================
if __name__ == "__main__":
    new_data_path = "kline_data/df_features.csv"


    # 定义seq_len和TARGET_COL的列表
    seq_lens = [400] * 6
    target_cols = ["close_up_0.2_t5", "close_up_0.2_t6", "close_up_0.2_t7", "close_up_0.2_t8", "close_up_0.25_t8", "close_up_0.25_t9"]

    predictors = []

    # 创建多个预测器实例
    for seq_len, target_col in zip(seq_lens, target_cols):
        models_dir = f"models/BTC-USDT-SWAP_1m_20230124_20241218_features_tail/{seq_len}_{target_col}"
        predictors.append(Predictor(models_dir, seq_len, target_col))

    pre_time_str = 0
    while True:
        current_time = time.time()
        current_time = pd.to_datetime(current_time, unit='s')
        start_time = time.time()

        # 这里可以控制每次循环间的间隔，避免不必要的频繁调用
        if start_time % 60 < 50:
            time.sleep(1)
            continue

        # 获取最新的数据
        feature_df = get_latest_data(max_candles=500)

        # 获取所有预测器的预测结果
        result_dfs = [predictor.predict(new_data_path) for predictor in predictors]

        # 为每个预测结果重命名列名，确保符合{seq_len}_{TARGET_COL}格式
        for i, df in enumerate(result_dfs):
            seq_len = seq_lens[i]
            target_col = target_cols[i]
            # 重命名列，将预测的目标列命名为 {seq_len}_{TARGET_COL}
            df.rename(columns={df.columns[1]: f'{seq_len}_{target_col}'}, inplace=True)

        # 合并所有预测结果
        result_df = merge_results(result_dfs)

        # 删除包含nan的行
        result_df = result_df.dropna()

        # 输出合并后的结果（或者保存到文件）
        print(result_df.head())
        result_df['pred_up_sum'] = result_df.filter(regex='pred.*up').sum(axis=1)

        # 计算包含 "down" 且包含 "pred" 的列的总和
        result_df['pred_down_sum'] = result_df.filter(regex='pred.*down').sum(axis=1)
        result_df['cha'] = result_df['pred_up_sum'] - result_df['pred_down_sum']
        result_df['he'] = result_df['pred_up_sum'] + result_df['pred_down_sum']
        # analyze_data(result_df)

        # pred_up_sum大于3.8的时候，买入

        # 获取最后一行数据
        latest_data = feature_df.iloc[-1]
        # 获取最后一行数据的价格
        latest_price = latest_data['close']
        last_time_str = latest_data['timestamp']
        # 同时获取result_df的最后一行数据
        latest_result = result_df.iloc[-1]
        #找到latest_result中包含"pred_final_prob"的列
        pred_cols = [col for col in latest_result.index if 'pred_final_prob' in col]
        # 获取最大的概率值

        if latest_result['pred_up_sum'] > 3.8 and last_time_str != pre_time_str:
            pre_time_str = last_time_str
            place_order(
                INST_ID,
                "buy",
                "limit",  # 限价单
                ORDER_SIZE,
                price=latest_price,  # 买入价格
                tp_price=latest_price + 200  # 止盈价格
            )
            print(f"{current_time} 最新价格：{latest_price}，最大latest_result['pred_up_sum']：{latest_result['pred_up_sum']}， {latest_result[pred_cols]}，下单成功！下单价格：{latest_price}，止盈价格：{latest_price + 200}")
        else:
            print(f"{current_time} 最新价格：{latest_price}，最大latest_result['pred_up_sum']：{latest_result['pred_up_sum']} {latest_result[pred_cols]}")
        release_funds(INST_ID, latest_price, 5)
        print(f"耗时：{time.time() - start_time}秒")
