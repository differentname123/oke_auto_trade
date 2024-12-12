import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import itertools
import multiprocessing
import os
def run():
    result_list = []
    # 遍历 models 下面所有的csv文件
    for root, dirs, files in os.walk('models'):
        for file in files:
            if file.endswith('.csv'):
                base_name = os.path.splitext(file)[0]
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                df['model'] = base_name
                df['score'] = df['1的准确率'] * df['预测为1的数量']
                result_list.append(df)

    # 合并所有结果
    result = pd.concat(result_list)
    result.to_csv('all_result.csv', index=False)
    print('已保存 all_result.csv')
if __name__ == "__main__":
    df = pd.read_csv('all_result.csv')
    run()