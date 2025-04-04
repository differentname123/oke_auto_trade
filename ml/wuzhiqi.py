import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from math import sqrt
import multiprocessing as mp
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


# ================================
# 1. 五子棋环境与状态表示
# ================================
class GomokuEnv:
    def __init__(self, board_size=9, win_length=5):
        self.board_size = board_size
        self.win_length = win_length
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1  # 1 表示先手，-1 表示后手
        self.game_over = False
        self.winner = 0  # 0 表示未分出胜负或平局
        return self.board

    def get_valid_moves(self):
        valid = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] == 0:
                    valid.append((r, c))
        return valid

    def step(self, move):
        r, c = move
        if self.board[r, c] != 0:
            raise ValueError("Invalid move!")
        self.board[r, c] = self.current_player
        if self.check_winner(r, c, self.current_player):
            self.game_over = True
            self.winner = self.current_player
        elif len(self.get_valid_moves()) == 0:
            self.game_over = True
            self.winner = 0
        else:
            self.current_player *= -1
        return self.board, self.game_over, self.winner

    def check_winner(self, row, col, player):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            # 正方向
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc
            # 反方向
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= self.win_length:
                return True
        return False


def board_to_tensor(board, current_player):
    """
    将棋盘转换为网络输入数据，两个通道：
      - 通道 0：当前玩家的棋子
      - 通道 1：对方的棋子
    """
    board = np.array(board)
    current_board = (board == current_player).astype(np.float32)
    opp_board = (board == -current_player).astype(np.float32)
    state = np.stack([current_board, opp_board], axis=0)  # shape: (2, board_size, board_size)
    return torch.tensor(state, dtype=torch.float32)


def is_terminal(board, win_length=5):
    board_size = board.shape[0]
    for r in range(board_size):
        for c in range(board_size):
            if board[r, c] != 0:
                player = board[r, c]
                directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
                for dr, dc in directions:
                    count = 1
                    rr, cc = r + dr, c + dc
                    while 0 <= rr < board_size and 0 <= cc < board_size and board[rr, cc] == player:
                        count += 1
                        rr += dr
                        cc += dc
                    rr, cc = r - dr, c - dc
                    while 0 <= rr < board_size and 0 <= cc < board_size and board[rr, cc] == player:
                        count += 1
                        rr -= dr
                        cc -= dc
                    if count >= win_length:
                        return True, player
    if np.sum(board == 0) == 0:
        return True, 0
    return False, None


def get_valid_moves_for_board(board):
    board_size = board.shape[0]
    moves = []
    for r in range(board_size):
        for c in range(board_size):
            if board[r, c] == 0:
                moves.append((r, c))
    return moves


def get_candidate_moves(board, radius=2):
    """
    限制搜索范围：仅考虑棋盘上已有棋子周围 radius 格范围内的空位，
    若棋盘未落子，则返回棋盘中央位置。
    """
    board_size = board.shape[0]
    stones = np.argwhere(board != 0)
    if len(stones) == 0:
        return [(board_size // 2, board_size // 2)]
    min_r = max(0, np.min(stones[:, 0]) - radius)
    max_r = min(board_size - 1, np.max(stones[:, 0]) + radius)
    min_c = max(0, np.min(stones[:, 1]) - radius)
    max_c = min(board_size - 1, np.max(stones[:, 1]) + radius)
    moves = []
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if board[r, c] == 0:
                moves.append((r, c))
    return moves


# ================================
# 2. 网络模型定义（包含残差块）
# ================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class GomokuNet(nn.Module):
    def __init__(self, board_size=9, num_channels=128, num_res_blocks=3):
        super(GomokuNet, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(2, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )
        # 策略头：输出 board_size * board_size 个 logits
        self.conv_policy = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * board_size * board_size, board_size * board_size)
        # 价值头：输出一个标量，使用 tanh 限幅在 [-1, 1]
        self.conv_value = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(board_size * board_size, 64)
        self.fc_value2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            x = block(x)
        # 策略头
        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = p.view(p.size(0), -1)
        p = self.fc_policy(p)
        # 价值头
        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_value1(v))
        v = torch.tanh(self.fc_value2(v))
        return p, v


# ================================
# 3. MCTS 模块（批量扩展、Dirichlet 噪声、树复用）
# ================================
class MCTSNode:
    def __init__(self, board, current_player, parent=None, prior=1.0):
        self.board = board.copy()
        self.current_player = current_player
        self.parent = parent
        self.children = {}  # 键为动作 (r, c)
        self.N = 0  # 访问次数
        self.W = 0.0  # 累计价值
        self.P = prior  # 先验概率
        self.is_expanded = False
        self.is_terminal, self.winner = is_terminal(self.board)


def select_child(node, c_puct):
    best_score = -float('inf')
    best_move = None
    best_child = None
    for move, child in node.children.items():
        Q = child.W / child.N if child.N > 0 else 0
        score = Q + c_puct * child.P * (np.sqrt(node.N) / (1 + child.N))
        if score > best_score:
            best_score = score
            best_move = move
            best_child = child
    return best_move, best_child


def query_inference(batch_tensor, inference_queue):
    """
    将输入 batch_tensor 中每个样本发送至中央推理服务器，并收集结果。
    """
    results = []
    # 确保输入在 CPU 上（将 tensor 转换为 numpy 数组传输）
    batch_tensor_cpu = batch_tensor.cpu()
    for i in range(batch_tensor_cpu.size(0)):
        parent_conn, child_conn = mp.Pipe()
        state_np = batch_tensor_cpu[i].numpy()
        inference_queue.put((child_conn, state_np))
        result = parent_conn.recv()  # 阻塞等待结果
        results.append(result)
    # 组合结果到 batch 中
    p_logits_list, v_pred_list = zip(*results)
    p_logits_batch = torch.tensor(np.stack(p_logits_list), dtype=torch.float32)
    v_pred_batch = torch.tensor(np.stack(v_pred_list), dtype=torch.float32)
    return p_logits_batch, v_pred_batch


def run_mcts(root, num_simulations, c_puct=1.0, add_dirichlet_noise=True,
             mcts_batch_size=16, network=None, inference_queue=None, device=None):
    """
    若 inference_queue 不为空，则通过中央推理服务器，否则直接使用局部网络（支持 AMP）。
    """
    # 如果需要扩展根节点且非终局，先扩展并添加 Dirichlet 噪声
    if add_dirichlet_noise and not root.is_expanded and not root.is_terminal:
        state_tensor = board_to_tensor(root.board, root.current_player).unsqueeze(0)
        if inference_queue is not None:
            p_logits, v = query_inference(state_tensor, inference_queue)
        else:
            state_tensor = state_tensor.to(device)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    p_logits, v = network(state_tensor)
        p_logits = p_logits[0].detach().cpu().numpy() if inference_queue is None else p_logits[0].numpy()
        v = v[0].item() if inference_queue is None else v[0].item()
        valid_moves = get_candidate_moves(root.board)
        valid_moves_indices = [m[0] * root.board.shape[0] + m[1] for m in valid_moves]
        p = F.softmax(torch.tensor(p_logits), dim=0).numpy()
        p_valid = np.zeros_like(p)
        if np.sum(p[valid_moves_indices]) > 0:
            p_valid[valid_moves_indices] = p[valid_moves_indices]
            p_valid = p_valid / np.sum(p_valid[valid_moves_indices])
        else:
            p_valid[valid_moves_indices] = 1.0 / len(valid_moves_indices)
        root.is_expanded = True
        for move in valid_moves:
            idx = move[0] * root.board.shape[0] + move[1]
            child_board = root.board.copy()
            child_board[move[0], move[1]] = root.current_player
            child = MCTSNode(child_board, -root.current_player, parent=root, prior=p_valid[idx])
            root.children[move] = child
        # 注入 Dirichlet 噪声
        epsilon = 0.25
        dir_alpha = 0.3
        noise = np.random.dirichlet([dir_alpha] * len(valid_moves))
        for i, move in enumerate(valid_moves):
            root.children[move].P = (1 - epsilon) * root.children[move].P + epsilon * noise[i]

    simulation_count = 0
    batch_nodes = []  # 存储待扩展节点及其路径

    while simulation_count < num_simulations:
        node = root
        path = [node]
        # 选择阶段
        while node.is_expanded and not node.is_terminal:
            move, next_node = select_child(node, c_puct)
            if next_node is None:
                break
            node = next_node
            path.append(node)
        if node.is_terminal:
            if node.winner == 0:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if (node.parent is not None and node.winner == node.parent.current_player) else -1.0
            simulation_count += 1
            for n in reversed(path):
                n.N += 1
                n.W += leaf_value
                leaf_value = -leaf_value
        else:
            batch_nodes.append((node, path))
            simulation_count += 1

        if (len(batch_nodes) >= mcts_batch_size) or (simulation_count == num_simulations and len(batch_nodes) > 0):
            state_list = []
            for candidate, _ in batch_nodes:
                state_list.append(board_to_tensor(candidate.board, candidate.current_player))
            state_batch = torch.stack(state_list)
            if inference_queue is not None:
                batch_p_logits, batch_v = query_inference(state_batch, inference_queue)
            else:
                state_batch = state_batch.to(device)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        batch_p_logits, batch_v = network(state_batch)
            if inference_queue is None:
                batch_p_logits_np = batch_p_logits.detach().cpu().numpy()
                batch_v_np = batch_v.detach().cpu().numpy().flatten()
            else:
                batch_p_logits_np = batch_p_logits.numpy()
                batch_v_np = batch_v.numpy().flatten()
            for i, (candidate, path) in enumerate(batch_nodes):
                logits = batch_p_logits_np[i]
                exp_logits = np.exp(logits - np.max(logits))
                p = exp_logits / np.sum(exp_logits)
                valid_moves = get_candidate_moves(candidate.board)
                valid_moves_indices = [m[0] * candidate.board.shape[0] + m[1] for m in valid_moves]
                p_valid = np.zeros_like(p)
                if np.sum(p[valid_moves_indices]) > 0:
                    p_valid[valid_moves_indices] = p[valid_moves_indices]
                    p_valid = p_valid / np.sum(p_valid[valid_moves_indices])
                else:
                    p_valid[valid_moves_indices] = 1.0 / len(valid_moves_indices)
                candidate.is_expanded = True
                for move in valid_moves:
                    idx = move[0] * candidate.board.shape[0] + move[1]
                    child_board = candidate.board.copy()
                    child_board[move[0], move[1]] = candidate.current_player
                    child = MCTSNode(child_board, -candidate.current_player, parent=candidate, prior=p_valid[idx])
                    candidate.children[move] = child
                leaf_value = batch_v_np[i]
                for n in reversed(path):
                    n.N += 1
                    n.W += leaf_value
                    leaf_value = -leaf_value
            batch_nodes = []
    return root


# ================================
# 4. 数据增强：旋转与翻转
# ================================
def augment_data(state, pi, z):
    board_size = int(np.sqrt(len(pi)))
    pi_board = pi.reshape(board_size, board_size)
    augmented = []
    for k in range(4):
        rotated_state = torch.rot90(state, k, dims=[1, 2])
        rotated_pi = np.rot90(pi_board, k)
        augmented.append((rotated_state, rotated_pi.flatten(), z))
        flipped_state = torch.flip(rotated_state, dims=[2])
        flipped_pi = np.fliplr(rotated_pi)
        augmented.append((flipped_state, flipped_pi.flatten(), z))
    return augmented


# ================================
# 5. 自我对弈（利用中央推理服务器进行批量预测）
# ================================
def self_play_game(inference_queue, num_mcts_simulations, temperature, augment=True):
    env = GomokuEnv(board_size=9, win_length=5)
    env.reset()
    game_data = []
    root_node = None
    while not env.game_over:
        if root_node is None or not np.array_equal(root_node.board, env.board):
            root_node = MCTSNode(env.board, env.current_player)
        # 使用中央推理服务器扩展节点
        root_node = run_mcts(root_node, num_mcts_simulations, c_puct=1.0,
                             add_dirichlet_noise=(root_node.parent is None),
                             mcts_batch_size=16,
                             inference_queue=inference_queue)
        pi = np.zeros(env.board_size * env.board_size)
        for move, child in root_node.children.items():
            index = move[0] * env.board_size + move[1]
            pi[index] = child.N
        if np.sum(pi) > 0:
            if temperature > 0:
                pi = pi ** (1.0 / temperature)
            pi = pi / np.sum(pi)
        else:
            pi = np.ones(env.board_size * env.board_size) / (env.board_size * env.board_size)
        state_tensor = board_to_tensor(env.board, env.current_player)
        game_data.append((state_tensor, pi, env.current_player))
        action_index = np.random.choice(np.arange(env.board_size * env.board_size), p=pi)
        row = action_index // env.board_size
        col = action_index % env.board_size
        env.step((row, col))
        if (row, col) in root_node.children:
            root_node = root_node.children[(row, col)]
            root_node.parent = None
        else:
            root_node = None
    outcome = env.winner
    training_data = []
    for state, pi, player in game_data:
        if outcome == 0:
            z = 0.0
        elif outcome == player:
            z = 1.0
        else:
            z = -1.0
        training_data.append((state, pi, z))
        if augment:
            training_data.extend(augment_data(state, pi, z))
    return training_data


# ================================
# 6. Replay Buffer（容量提高至 50000）
# ================================
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = []

    def add(self, data):
        self.buffer.extend(data)
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# ================================
# 7. 模型评估（新模型与最佳模型各执先手，各自对局数量增加，并计算 95% CI）
# ================================
def simulate_game_evaluation(new_first, new_network, best_network, device, num_mcts_simulations):
    env = GomokuEnv(board_size=9, win_length=5)
    env.reset()
    while not env.game_over:
        if new_first:
            network_dict = {1: new_network, -1: best_network}
        else:
            network_dict = {1: best_network, -1: new_network}
        current_network = network_dict[env.current_player]
        root = MCTSNode(env.board, env.current_player)
        root = run_mcts(root, num_mcts_simulations, c_puct=1.0, add_dirichlet_noise=False,
                        network=current_network, inference_queue=None, device=device)
        best_move = None
        max_visits = -1
        for move, child in root.children.items():
            if child.N > max_visits:
                max_visits = child.N
                best_move = move
        env.step(best_move)
    return env.winner


def evaluate_model(new_network, best_network, device, num_games=40, num_mcts_simulations=400):
    games_per_role = num_games // 2
    new_wins = 0
    for new_first in [True, False]:
        for _ in range(games_per_role):
            winner = simulate_game_evaluation(new_first, new_network, best_network, device, num_mcts_simulations)
            if new_first and winner == 1:
                new_wins += 1
            elif not new_first and winner == -1:
                new_wins += 1
    total_games = games_per_role * 2
    win_rate = new_wins / total_games
    se = sqrt(win_rate * (1 - win_rate) / total_games)
    ci_low = win_rate - 1.96 * se
    ci_high = win_rate + 1.96 * se
    print(f"Evaluation: win rate = {win_rate:.2f} (95% CI: [{ci_low:.2f}, {ci_high:.2f}])")
    return win_rate


# ================================
# 8. 中央推理服务器
# ================================
def inference_server_main(model, inference_queue, stop_event):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("Inference server started on device:", device)
    while not stop_event.is_set() or not inference_queue.empty():
        requests = []
        try:
            req = inference_queue.get(timeout=0.1)
            requests.append(req)
        except Exception:
            continue
        while True:
            try:
                req = inference_queue.get_nowait()
                requests.append(req)
            except Exception:
                break
        if requests:
            inputs = []
            conns = []
            for conn, state_np in requests:
                inputs.append(torch.tensor(state_np, dtype=torch.float32))
                conns.append(conn)
            batch = torch.stack(inputs).to(device)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    p_logits, v_pred = model(batch)
            p_logits_np = p_logits.detach().cpu().numpy()
            v_pred_np = v_pred.detach().cpu().numpy()
            for i, conn in enumerate(conns):
                conn.send((p_logits_np[i], v_pred_np[i]))
                conn.close()
    print("Inference server exiting.")


# ================================
# 辅助函数：混合采样（保证 50% 来自当前 epoch 自对弈产生的新数据）
# ================================
def sample_mixed(batch_size, new_data, buffer_data, new_ratio=0.5):
    num_new = int(batch_size * new_ratio)
    if len(new_data) >= num_new:
        new_samples = random.sample(new_data, num_new)
    else:
        new_samples = new_data.copy()
    num_old = batch_size - len(new_samples)
    if len(buffer_data) >= num_old:
        old_samples = random.sample(buffer_data, num_old)
    else:
        old_samples = buffer_data.copy()
    return new_samples + old_samples


# ================================
# 9. 主训练流程（自对弈 + 学习率衰减 + 正则化 + 温度衰减 + 定期评估）
# ================================
def train_model(num_epochs=1000,
                games_per_epoch=20,
                num_mcts_simulations=400,
                initial_temperature=1.0,
                batch_size=64,
                lr=0.001,
                train_steps_per_epoch=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using training device:", device)
    network = GomokuNet(board_size=9, num_channels=128, num_res_blocks=3).to(device)
    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler()
    replay_buffer = ReplayBuffer(capacity=500000)
    best_network = copy.deepcopy(network)
    best_network.to(device)
    eval_interval = 20

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch} started.")
        if epoch < 10:
            temperature = 1.0
        else:
            temperature = 0.0
        print(f"Current temperature: {temperature:.4f}")

        # 使用 Manager 创建共享的 Queue 与 Event，解决 Queue 对象传递的问题
        manager = mp.Manager()
        inference_queue = manager.Queue()
        stop_event = manager.Event()

        frozen_model = copy.deepcopy(network).to(device)
        frozen_model.eval()
        print("Starting centralized inference server for self-play.")
        inference_server_process = mp.Process(target=inference_server_main,
                                              args=(frozen_model, inference_queue, stop_event))
        inference_server_process.start()

        print(f"Starting {games_per_epoch} self-play games in parallel.")
        new_data_total = []
        pool = mp.Pool(processes=min(games_per_epoch, mp.cpu_count()))
        results = [pool.apply_async(self_play_game,
                                    args=(inference_queue, num_mcts_simulations, temperature, True))
                   for _ in range(games_per_epoch)]
        for r in results:
            game_data = r.get()
            new_data_total.extend(game_data)
        pool.close()
        pool.join()
        replay_buffer.add(new_data_total)
        print(f"Self-play games completed. Replay buffer size: {len(replay_buffer)}")

        stop_event.set()
        inference_server_process.join()
        print("Centralized inference server stopped.")

        # ---------------------------
        # 训练阶段：执行多个 mini-batch 梯度更新
        # 采用混合采样策略：50% 来自当前 epoch 新生成的数据，50% 随机从 replay buffer 采样
        # ---------------------------
        total_loss_value = 0.0
        loss_count = 0
        for step in range(train_steps_per_epoch):
            if len(replay_buffer.buffer) < batch_size:
                break
            minibatch = sample_mixed(batch_size, new_data_total, replay_buffer.buffer, new_ratio=0.5)
            states, target_pis, target_zs = zip(*minibatch)
            batch_state = torch.stack(states).to(device)
            batch_pi = torch.tensor(target_pis, dtype=torch.float32, device=device)
            batch_z = torch.tensor(target_zs, dtype=torch.float32, device=device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                p_logits, v_pred = network(batch_state)
                log_probs = F.log_softmax(p_logits, dim=1)
                policy_loss = -torch.mean(torch.sum(batch_pi * log_probs, dim=1))
                value_loss = F.mse_loss(v_pred.squeeze(), batch_z)
                loss = policy_loss + value_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss_value += loss.item()
            loss_count += 1
        avg_loss = total_loss_value / loss_count if loss_count > 0 else 0.0

        scheduler.step()
        print(f"Epoch {epoch} complete. Replay Buffer: {len(replay_buffer)}, "
              f"Avg Loss: {avg_loss:.4f}, Temperature: {temperature:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        torch.save(network.state_dict(), f"gomoku_model_epoch.pth")

        if epoch % eval_interval == 0:
            eval_simulations = num_mcts_simulations
            print("Starting evaluation.")
            win_rate = evaluate_model(network, best_network, device, num_games=100,
                                      num_mcts_simulations=eval_simulations)
            if win_rate > 0.55:
                best_network.load_state_dict(network.state_dict())
                torch.save(network.state_dict(), "gomoku_best_model.pth")
                print("New best model set.")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    train_model(num_epochs=1000,
                games_per_epoch=30,
                num_mcts_simulations=400,
                initial_temperature=1.0,
                batch_size=64,
                lr=0.001,
                train_steps_per_epoch=200)