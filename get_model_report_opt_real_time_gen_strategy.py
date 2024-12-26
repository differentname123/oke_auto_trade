import json
import os
import traceback
from multiprocessing import Pool
from collections import defaultdict

import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from lightgbm import LGBMClassifier
from tqdm import tqdm

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

def gen_score(result_list, min_proportion=0.0):
    # 假设 result_list 是一个列表，其中每个元素都是一个包含指标的字典
    # 您需要根据自己的实际情况替换下面的实现
    # 这里我们假设每个元素都有一个键为 'profit' 的值，我们将所有的 profit 相加作为 score
    score = 0
    count = 40

    for result in result_list:
        count -= 1
        if count < 0:
            break
        if result['proportion'] >= min_proportion:
            score += float(result['proportion']) * float(result['percentile'])
    return score

def gen_score_op(result_list, base_percentile=1, min_proportion=0.0):
    # 按照相较于基准的成功率来排序
    score = 0
    count = 20
    for result in result_list:
        if result['percentile'] == base_percentile:
            base_proportion = float(result['proportion'])
    for result in result_list:
        count -= 1
        if count < 0:
            break
        if result['proportion'] >= min_proportion:
            score += (float(result['proportion']) - base_proportion) / base_proportion * float(result['percentile'])
    return score

def process_strategy_results(input_file, output_file):
    # 读取 strategy_result_back.json 文件
    with open(input_file, 'r') as f:
        strategy_results = json.load(f)

    # 按照 'target_column' 分组
    grouped_results = {}
    for entry in strategy_results:
        target_column = entry['target_column']
        if target_column not in grouped_results:
            grouped_results[target_column] = []
        grouped_results[target_column].append(entry)

    # 对每组中的每个 entry 调用 gen_score，并选择最高的 score
    final_results = []
    for target_column, entries in grouped_results.items():
        # if '_up_0.3_' not in target_column:
        #     continue
        for entry in entries:
            # 对每个 entry 的 result_list 调用 gen_score
            score = gen_score(entry['result_list'])
            entry['score'] = score  # 将 score 添加到 entry 中
        # sorted_entries = sorted(entries, key=lambda x: x['score'], reverse=True)
        # 按照 score 排序，选择最高的
        best_entry = max(entries, key=lambda x: x['score'])
        final_results.append(best_entry)
    # 将final_results按照score排序
    final_results = sorted(final_results, key=lambda x: x['score'], reverse=True)
    # 将最终的筛选结果写入 final_strategy_result.json
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=4)

    print(f"筛选完成，结果已保存到 {output_file}")


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
        if (time.time() - last_updated) <= 50:
            print(f"保留订单 {order_id}，订单价格：{order['px']}，最新价格：{latest_price}，差距：{price_diff}，时间间隔在1分钟内。")
            continue

        print(
            f"取消订单 {order_id}，订单价格：{order['px']}，最新价格：{latest_price}，差距：{price_diff}"
        )
        cancel_order(inst_id, order_id)


def compute_intersection_proportion(result_df, sort_columns, percentiles, target_column, min_proportion=0.7):
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
        threshold_dict = {}

        # 按照每个排序字段排序并取前 percentile 部分
        for col in sort_columns:
            # 排序并取前 percentile 部分的布尔索引
            sorted_df = result_df.sort_values(by=col, ascending=False)
            threshold_idx = int(len(sorted_df) * percentile)
            current_mask = sorted_df.index[:threshold_idx]
            threshold_dict[col] = sorted_df[col].iloc[threshold_idx - 1]

            # 对交集进行更新，取交集
            intersection_mask &= result_df.index.isin(current_mask)


        # 通过交集布尔索引，获取交集数据
        subset_df = result_df[intersection_mask]

        # 计算 '400_close_up_0.25_t9' 为 1 的占比
        proportion = (subset_df[target_column] == 1).mean()

        # 存储结果
        proportions[percentile] = proportion
        # if proportion > min_proportion or percentile>=0.5:
        result_list.append({
            'percentile': percentile,
            'proportion': proportion,
            'threshold':threshold_dict,
            'count': len(subset_df)
        })

    return result_list


def analyze_data(result_df, target_column='400_close_down_0.25_t8', sort_columns=['pred_down_sum']):
    # 删除result_df的后60行
    result_df = result_df.iloc[:-60]
    sort_columns = sort_columns # 排序字段列表
    percentiles = [x / 200 for x in range(1, 50)]  # 分段比例列表
    percentiles.append(0.5)
    percentiles.append(1)
    target_column = target_column  # 目标列

    proportions = compute_intersection_proportion(result_df, sort_columns, percentiles, target_column)

    return proportions


def merge_results(results):
    # 合并多个预测结果并重命名列，防止列名冲突
    merged_df = results[0]
    for result in results[1:]:
        merged_df = pd.merge(merged_df, result, on='timestamp', how='outer')
    return merged_df

# ===================== 使用示例 =====================
def extract_model_info(model_dir_name):
    """从模型目录名中提取模型信息"""
    try:
        split_list = model_dir_name.split('_')
        seq_len = int(split_list[0])
        side = split_list[2]
        profit = float(split_list[3])
        period = int(split_list[4].replace('t', ''))
        target_col = '_'.join(split_list[1:])
        return seq_len, target_col, side, profit, period
    except Exception as e:
        print(f"文件夹名字不符合规范：{model_dir_name}，错误：{e}")
        return None

def collect_model_infos(model_path):
    """遍历模型目录，收集模型信息"""
    model_infos = []
    for root, dirs, _ in os.walk(model_path):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if len(os.listdir(dir_path)) != 8:
                continue  # 模型文件不完整，跳过
            model_info = extract_model_info(dir_name)
            if model_info:
                model_infos.append({
                    'dir_name': dir_name,
                    'dir_path': dir_path,
                    'seq_len': model_info[0],
                    'target_col': model_info[1],
                    'side': model_info[2],
                    'profit': model_info[3],
                    'period': model_info[4],
                })
    return model_infos
def predictor_predict(args):
    """单个预测器的预测函数，用于多进程调用"""
    models_dir = args['models_dir']
    seq_len = args['seq_len']
    target_col = args['target_col']
    data_path = args['data_path']
    try:
        start_time = time.time()
        predictor = Predictor(models_dir, seq_len, target_col)
        df = predictor.predict(data_path)
        # 重命名预测列
        pred_col_name = f"{seq_len}_{target_col}"
        df.rename(columns={df.columns[1]: pred_col_name}, inplace=True)
        return df
    except Exception as e:
        print(f"预测 {seq_len}_{target_col} 失败，错误：{e}")
        return None

def generate_predictions(model_infos, data_path):
    """获取所有预测器的预测结果，使用多进程"""
    result_dfs = []
    tasks = []
    for info in model_infos:
        args = {
            'models_dir': info['dir_path'],
            'seq_len': info['seq_len'],
            'target_col': info['target_col'],
            'data_path': data_path
        }
        tasks.append(args)

    max_workers = 5
    pool_size = min(max_workers, 5)

    start_time = time.time()
    with Pool(processes=pool_size) as pool:
        results = pool.map(predictor_predict, tasks)
    total_time = time.time() - start_time
    print(f"所有预测器的预测结果已生成，耗时：{total_time:.2f} 秒")

    count = 0
    for df in results:
        count += 1
        if df is not None:
            result_dfs.append(df)
        else:
            print(f"警告：第 {count} 个预测器的预测结果为空")
    return result_dfs

def filter_model_infos(model_infos, target_side, target_profit, target_period, min_profit=-0.1, max_profit=0.1, min_period=-1, max_period=1, min_seq_len=100, max_seq_len=500):
    """根据条件筛选模型信息"""
    filtered_infos = []
    for info in model_infos:
        if info['seq_len'] < min_seq_len:
            continue
        if info['seq_len'] > max_seq_len:
            continue
        if info['side'] != target_side:
            continue
        if (info['profit'] - target_profit) > max_profit:
            continue
        if (info['profit'] - target_profit) < min_profit:
            continue
        if (info['period'] - target_period) > max_period:
            continue
        if (info['period'] - target_period) < min_period:
            continue
        filtered_infos.append(info)
    return filtered_infos


def process_target_column(args):
    target_column, all_model_infos, parameter_combinations, result_df = args
    strategy_results = []

    # 记录开始处理的时间
    start_time = time.time()

    # Extract target column parameters
    split_col = target_column.split('_')
    target_side = split_col[1]
    target_profit = float(split_col[2])
    target_period = int(split_col[3].replace('t', ''))

    for min_profit, max_profit, min_period, max_period, min_seq_len, max_seq_len in parameter_combinations:
        # Filter model infos based on the parameters
        filtered_model_infos = filter_model_infos(
            all_model_infos, target_side, target_profit, target_period,
            min_profit, max_profit, min_period, max_period, min_seq_len, max_seq_len
        )

        if not filtered_model_infos:
            continue  # Skip if no models match the criteria

        pred_final_prob_cols = []
        for info in filtered_model_infos:
            col_name = f"pred_final_prob_{info['seq_len']}_{info['target_col']}"
            pred_final_prob_cols.append(col_name)

        # Check if columns exist in result_df
        missing_cols = [col for col in pred_final_prob_cols if col not in result_df.columns]
        if missing_cols:
            print(f"警告：以下列在 result_df 中不存在，已忽略：{missing_cols}")
            # Remove missing columns
            pred_final_prob_cols = [col for col in pred_final_prob_cols if col in result_df.columns]

        if not pred_final_prob_cols:
            print(f"错误：没有可用的预测列进行计算，跳过该 target_column：{target_column}")
            continue

        # Calculate target_column
        temp_target_col_pred_cols = [col for col in pred_final_prob_cols if target_column in col]
        if not temp_target_col_pred_cols:
            # print(f"错误：找不到包含 {target_column} 的列，跳过该 target_column，Parameters: min_profit={min_profit}, max_profit={max_profit}, min_period={min_period}, max_period={max_period} min_seq_len={min_seq_len}, max_seq_len={max_seq_len}")
            continue

        # Calculate sum of prediction probabilities
        result_df['pred_sum'] = result_df[pred_final_prob_cols].sum(axis=1)

        temp_seq_len = int(temp_target_col_pred_cols[0].split('_')[3])
        sort_columns = ['pred_sum']

        # Analyze data
        result_list = analyze_data(
            result_df,
            target_column=f"{temp_seq_len}_{target_column}",
            sort_columns=sort_columns
        )

        # Save results with parameter identifiers
        temp_dict = {
            'min_profit': min_profit,
            'max_profit': max_profit,
            'min_period': min_period,
            'max_period': max_period,
            'min_seq_len': min_seq_len,
            'max_seq_len': max_seq_len,
            'pred_final_prob_cols_len': len(pred_final_prob_cols),
            'target_column': target_column,
            'result_list': result_list,
            'pred_final_prob_cols': pred_final_prob_cols
        }
        strategy_results.append(temp_dict)

    # 计算处理时间
    elapsed_time = time.time() - start_time
    print(f"完成处理 Target Column: {target_column}，耗时 {elapsed_time:.2f} 秒")

    return strategy_results


def main():
    from itertools import product
    import time

    data_path = "kline_data/df_features.csv"
    model_path = 'models/BTC-USDT-SWAP_1m_20230124_20241218_features_tail/'
    all_result_df_path = 'all_result_df.csv'

    # 记录程序开始时间
    total_start_time = time.time()

    # Collect all model infos
    all_model_infos = collect_model_infos(model_path)

    # Try to load existing prediction results
    if os.path.exists(all_result_df_path):
        result_df = pd.read_csv(all_result_df_path)
        print(f"已加载已有的预测结果，共 {len(result_df)} 行")
    else:
        feature_df = get_latest_data(max_candles=8000)

        # Get predictions from all predictors
        result_dfs = generate_predictions(all_model_infos, data_path)

        # Merge all prediction results
        result_df = merge_results(result_dfs)

        # Drop rows with NaN
        result_df.dropna(inplace=True)

        # Save result_df to CSV file
        result_df.to_csv(all_result_df_path, index=False)

    # Get all unique target columns
    target_column_set = set(info['target_col'] for info in all_model_infos)
    target_column_list = list(target_column_set)

    # 定义总的任务数
    total_tasks = len(target_column_list)
    print(f"总任务数：{total_tasks}")

    # Define parameter values
    min_profit_values = [-0.1, -0.05, 0]
    max_profit_values = [0.1, 0.05, 0]
    min_period_values = [-3, -2, -1, 0]
    max_period_values = [3, 2, 1, 0]
    min_seq_len_values = [0, 100]
    max_seq_len_values = [500, 600]

    # Generate parameter combinations, ensuring min_profit <= max_profit and min_period <= max_period
    parameter_combinations = []
    for min_profit, max_profit in product(min_profit_values, max_profit_values):
        if min_profit > max_profit:
            continue
        for min_period, max_period in product(min_period_values, max_period_values):
            if min_period > max_period:
                continue
            for min_seq_len, max_seq_len in product(min_seq_len_values, max_seq_len_values):
                if min_seq_len > max_seq_len:
                    continue
                parameter_combinations.append((min_profit, max_profit, min_period, max_period, min_seq_len, max_seq_len))
    print(f"共 {len(parameter_combinations)} 种参数组合")
    # Remove duplicate parameter combinations
    parameter_combinations = list(set(parameter_combinations))
    print(f"去除重复后，共 {len(parameter_combinations)} 种参数组合")

    # Prepare arguments for multiprocessing
    args_list = []
    for target_column in target_column_list:
        args = (target_column, all_model_infos, parameter_combinations, result_df)
        args_list.append(args)

    # Use multiprocessing pool to process each target_column with a progress bar
    strategy_results = []
    with Pool(processes=5) as pool:
        # 使用 tqdm 的进度条来跟踪进度，total 参数设置为任务总数
        for res in tqdm(pool.imap_unordered(process_target_column, args_list), total=total_tasks):
            strategy_results.extend(res)

    # 计算总的执行时间
    total_elapsed_time = time.time() - total_start_time
    print(f"所有任务完成，总耗时 {total_elapsed_time:.2f} 秒")

    # Save strategy results
    with open('strategy_result_back.json', 'w') as f:
        json.dump(strategy_results, f, indent=4)

def analyze_strategy_result():
    # 读取策略结果
    with open('strategy_result.json', 'r') as f:
        strategy_results = json.load(f)
    final_list = []
    # 遍历策略结果
    for result in strategy_results:
        target_column = result['target_column']
        print(f"Target Column: {target_column}")
        for item in result['result_list']:
            # 只保留proportion大于0.7的
            if item['proportion'] > 0.7:
                print(f"Percentile: {item['percentile']:.2f}, Proportion: {item['proportion']:.2f}, Count: {item['count']}, Threshold: {item['threshold']}")

def load_good_model_infos():
    # 读取策略结果
    with open('final_strategy_result.json', 'r') as f:
        strategy_results = json.load(f)


def order():
    data_path = "kline_data/df_features.csv"
    strategy_results = top_n_scores_by_group('final_strategy_result.json', 2)
    dir_name_list = []
    final_strategy_results = []
    # 遍历策略结果
    for result in strategy_results:
        pred_final_prob_cols = result['pred_final_prob_cols']
        final_strategy_results.append(result)
        for col in pred_final_prob_cols:
            seq_col = col.split('prob_')[1]
            dir_name_list.append(seq_col)
    strategy_results = final_strategy_results
    # 对seq_col_list去重
    dir_name_list = list(set(dir_name_list))
    seq_lens = []
    target_cols = []

    all_model_infos = []
    for dir_name in dir_name_list:
        seq_len = dir_name.split('_')[0]
        seq_lens.append(seq_len)
        # target_col是以'_'分割除了seq_len的部分
        target_col = '_'.join(dir_name.split('_')[1:])
        target_cols.append(target_col)
        model_info = extract_model_info(dir_name)

        all_model_infos.append({
            'dir_name': dir_name,
            'dir_path': f"models/BTC-USDT-SWAP_1m_20230124_20241218_features_tail/{seq_len}_{target_col}",
            'seq_len': model_info[0],
            'target_col': model_info[1],
            'side': model_info[2],
            'profit': model_info[3],
            'period': model_info[4],
        })
    print(f"all_model_infos长度：{len(all_model_infos)}")
    offset = 20
    while True:
        try:
            latest_price = 100000
            start_time = time.time()
            # if start_time % 60 < 58:
            #     time.sleep(1)
            #     continue
            # all_model_infos = all_model_infos[:10]
            feature_df = get_latest_data(max_candles=1500)

            result_df = pd.read_csv('single_result_df.csv')

            # result_df = generate_predictions(all_model_infos, data_path)
            # # 合并所有预测结果
            # result_df = merge_results(result_df)
            #
            # # 删除包含 NaN 的行
            # result_df.dropna(inplace=True)
            # # 取最新的一行
            # # 将result_df保存为CSV文件
            # result_df.to_csv('single_result_df.csv', index=False)

            final_result_df = gen_detail_info(result_df, strategy_results)
            # 将analysis_result移动到第一列
            final_result_df = final_result_df[['analysis_result'] + [col for col in final_result_df.columns if col != 'analysis_result']]
            # 获取最后一行的analysis_result
            final_result = final_result_df.iloc[-1]['analysis_result']
            for key in final_result.keys():
                if key == 'timestamp':
                    continue
                if final_result[key]['proportion'] >= 0.7 and final_result[key]['percentile'] < 0.5:
                    latest_data = feature_df.iloc[-1]
                    latest_price = latest_data['close']
                    seq_len, target_col, side, profit, period = extract_model_info(f'100_{key}')
                    if side == 'down':
                        # 卖出
                        result = place_order(
                            INST_ID,
                            "sell",
                            "limit",
                            ORDER_SIZE,
                            price=latest_price + offset,
                            tp_price=latest_price - profit*1000 + offset  # 设置止盈价格
                        )
                    else:
                        # 买入
                        result = place_order(
                            INST_ID,
                            "buy",
                            "limit",
                            ORDER_SIZE,
                            price=latest_price - offset,
                            tp_price=latest_price + profit*1000 - offset  # 设置止盈价格
                        )
                    print(f"下单成功：{key} {result} final_result[key]: {final_result[key]}")
                    break
            release_funds(INST_ID, latest_price, 5)

            print(final_result)
            final_result['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            # 将timestamp移动到第一列
            final_result = {k: final_result[k] for k in ['timestamp'] + list(final_result.keys())}

            # 输出当前时间
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            # 输出result_df的最后一行
            print(result_df.iloc[-1])
            print("耗时：", time.time() - start_time)
            # 将final_result增量写入到文件
            with open("final_result.txt", "a") as f:
                f.write(str(final_result) + "\n")
        except Exception as e:
            print(f"错误：{e}")
            traceback.print_exc()
            continue


def gen_detail_info(result_df, strategy_results):
    from copy import deepcopy

    def process_row(row):
        final_result_dict = {}
        columns = row.index

        for result in strategy_results:
            pred_final_prob_cols = result['pred_final_prob_cols']
            history_result_list = result['result_list']
            target_column = result['target_column']

            # 检查列是否存在于 row 中
            missing_cols = [col for col in pred_final_prob_cols if col not in columns]
            if missing_cols:
                print(f"警告：以下列在 row 中不存在，已忽略：{missing_cols}")
                # 从列表中移除缺失的列
                pred_final_prob_cols = [col for col in pred_final_prob_cols if col in columns]

            if not pred_final_prob_cols:
                print(f"错误：没有可用的预测列进行计算，跳过该 target_column：{target_column}")
                continue

            row_pred_sum = row[pred_final_prob_cols].sum()

            result_found = False
            for original_item in history_result_list:
                # 深拷贝 original_item，以避免修改原始数据
                item = deepcopy(original_item)
                if item['threshold']['pred_sum'] < row_pred_sum:
                    item['pred_sum'] = row_pred_sum
                    final_result_dict[target_column] = item
                    result_found = True
                    break
            if not result_found:
                # 当没有匹配的项时，可以根据需要处理
                pass

        filtered_items = {key: value for key, value in final_result_dict.items() if value['proportion'] >= 0.5 and value['percentile'] < 0.5}

        # 2. 按 percentile 升序排序
        sorted_items = sorted(filtered_items.items(), key=lambda item: item[1]['percentile'], reverse=False)

        # 3. 将排序后的结果重新转换为字典
        final_result_dict = dict(sorted_items)

        return final_result_dict

    # 对每一行应用 process_row 函数，并将结果存储在新的列中
    result_df['analysis_result'] = result_df.apply(process_row, axis=1)

    return result_df


def top_n_scores_by_group(json_file_path, n):
    # result_list = []
    # # 读取策略结果
    # with open('strategy_result.json', 'r') as f:
    #     strategy_results = json.load(f)
    # dir_name_list = []
    # final_strategy_results = []
    # # 遍历策略结果
    # for result in strategy_results:
    #     pred_final_prob_cols = result['pred_final_prob_cols']
    #     if float(result['score']) > 0.01:
    #         result_list.append(result)
    #
    # return result_list


    # 读取 JSON 文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # 分组字典，键为分组依据 (第二和第三个元素)，值为该组的数据列表
    groups = defaultdict(list)

    # 遍历数据并进行分组
    for item in data:
        target_column = item['target_column']

        # 分割 target_column 以提取第二和第三个元素
        target_parts = target_column.split('_')
        group_key = (target_parts[1], target_parts[2])  # 以 (up/down, 0.15/0.2等) 为分组依据

        # 将数据按分组存储
        groups[group_key].append(item)

    # 准备返回的结果
    result = []

    # 对每个分组按 score 降序排序，并取前 n 个
    for group_key, items in groups.items():
        # 按 score 排序，获取前 n 个数据
        top_n_items = sorted(items, key=lambda x: x['score'], reverse=True)[:n]
        result.extend(top_n_items)

    return result


if __name__ == "__main__":
    # process_strategy_results('strategy_result_back.json', 'final_strategy_result.json')

    order()
    # analyze_strategy_result()
    # main()