# k层可预测性（L_k）分析脚本
#
# 指标定义：L_k（k层可预测性）= lift（相对多数基线的胜算比）
#   - L_k > 1: 该层可约（存在可预测结构）
#   - L_k ≈ 1: 无结构（随机噪声）
#   - L_k < 1 或极低: 该层不可约（强边界依赖或混沌）
#
# 流程：
#   1. 在 Rule 110 上用块熵 + 连续低熵块判定活跃区，得到统一动态高度
#   2. 裁剪所有规则（110/30/32/184）到该高度
#   3. 在固定大小区域（256×256）上采样，计算 L1–L30
#   4. 绘制箱线图展示各层级 L_k 分布
#   5. 选择极差最大层级进行 Benford 分析
#
# 目的：用 L_k 的"鼓包"刻画计算层次性（多尺度的可约结构），
#      并观察 Benford 现象是否在这些层上更强

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chisquare

# 设置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def evolve_rule(rule_bits, width=768, steps=900, p=0.5, seed=0):
    rng = np.random.default_rng(seed)
    grid = np.zeros((steps, width), dtype=np.uint8)
    grid[0] = (rng.random(width) < p).astype(np.uint8)
    table = np.array(rule_bits, dtype=np.uint8)  # for 000..111
    for t in range(steps-1):
        L = np.roll(grid[t], 1)
        C = grid[t]
        R = np.roll(grid[t], -1)
        idx = (L<<2) | (C<<1) | R
        grid[t+1] = table[idx]
    return grid

def calculate_block_entropy(block_data, block_height, block_width, ngram_height=2, ngram_width=2):
    """
    计算块熵（N-gram Entropy）
    
    参数:
        block_data: 二维数组 (block_height, block_width)
        block_height: 块的高度
        block_width: 块的宽度
        ngram_height: 内部小滑块的高度（默认2）
        ngram_width: 内部小滑块的宽度（默认2）
    
    返回:
        熵值（float）
    """
    # 统计N元块的频率
    pattern_counts = {}
    total_patterns = 0
    
    # 在矩阵中滑动ngram_height×ngram_width的小窗口
    for i in range(block_height - ngram_height + 1):
        for j in range(block_width - ngram_width + 1):
            # 提取ngram_height×ngram_width块
            pattern = ''
            for di in range(ngram_height):
                for dj in range(ngram_width):
                    pattern += str(block_data[i + di, j + dj])
            
            # 统计该模式
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            total_patterns += 1
    
    # 如果没有提取到任何模式，返回0
    if total_patterns == 0:
        return 0.0
    
    # 计算香农熵
    entropy = 0.0
    for count in pattern_counts.values():
        p = count / total_patterns
        if p > 0:
            entropy += -p * np.log2(p)
    
    return entropy

def compute_dynamic_height(grid, block_height=20, ngram_height=2, ngram_width=2, 
                          num_intervals=9, cutoff_interval=1, consecutive_count=3):
    """
    动态计算合适的CA高度：找到连续N个处于低熵区间（不活跃）的块，在此之前截断
    
    参数:
        grid: CA时空图 (steps, width)
        block_height: 块的高度（默认20）
        ngram_height: 内部小滑块高度（默认2）
        ngram_width: 内部小滑块宽度（默认2）
        num_intervals: 熵值分区数量（默认9）
        cutoff_interval: 截断区间编号（1-9），遇到此区间及以下就截断（默认1=最低熵区间）
        consecutive_count: 需要连续多少个低熵块才截断（默认3）
    
    返回:
        合适的高度值（int），即第一组连续低熵块的起始位置（不包括这些块）
    """
    steps, width = grid.shape
    
    # 计算所有可能的块熵值
    entropy_values = []
    block_positions = []  # 记录每个块的起始位置
    
    for y in range(0, steps - block_height + 1, block_height):
        block = grid[y:y+block_height, :]
        entropy = calculate_block_entropy(block, block_height, width, ngram_height, ngram_width)
        entropy_values.append(entropy)
        block_positions.append(y)
    
    if len(entropy_values) == 0:
        return steps  # 如果没有计算出熵值，返回全部步数
    
    # 找出熵值的范围
    min_entropy = np.min(entropy_values)
    max_entropy = np.max(entropy_values)
    entropy_range = max_entropy - min_entropy
    
    if entropy_range == 0:
        return steps  # 如果熵值都相同，返回全部步数
    
    # 计算截断区间的上边界阈值
    # cutoff_interval=1表示第1个区间（最低熵），其上边界是 min + 1/num_intervals * range
    cutoff_threshold = min_entropy + cutoff_interval / num_intervals * entropy_range
    
    # 找到连续N个处于截断区间或更低的块
    consecutive_low = 0  # 连续低熵块计数器
    for i, entropy in enumerate(entropy_values):
        if entropy < cutoff_threshold:
            consecutive_low += 1
            if consecutive_low >= consecutive_count:
                # 找到了连续N个低熵块，返回第一个低熵块的起始位置
                first_low_block_idx = i - consecutive_count + 1
                if first_low_block_idx <= 0:
                    # 如果从开始就是低熵，至少返回一个块的高度
                    return block_height
                return block_positions[first_low_block_idx]
        else:
            # 遇到非低熵块，重置计数器
            consecutive_low = 0
    
    # 如果没有找到连续N个低熵块，说明一直都很活跃，返回全部步数
    return steps

RULE110 = [0,1,1,1,0,1,1,0]  # Class IV (图灵完备)
RULE30  = [0,1,1,1,1,0,0,0]  # Class III (混沌)
RULE32  = [0,0,0,0,0,1,0,0]  # Class I (简单收敛)
RULE184 = [0,0,0,1,1,1,0,1]  # Class II (交通流模型)

# 生成随机种子（用于可复现性）
rng_main = np.random.default_rng()
seed110 = rng_main.integers(0, 2**31)
seed30 = rng_main.integers(0, 2**31)
seed32 = rng_main.integers(0, 2**31)
seed184 = rng_main.integers(0, 2**31)

print("\n" + "="*70)
print("随机种子信息 (用于复现结果)")
print("="*70)
print(f"Rule 110 种子: {seed110}")
print(f"Rule 30  种子: {seed30}")
print(f"Rule 32  种子: {seed32}")
print(f"Rule 184 种子: {seed184}")
print("="*70)

# 第一步：生成较大的初始grid（8000步）
print("\n" + "="*70)
print("第一步：生成初始CA时空图（8000步）")
print("="*70)
initial_steps = 8000
grid110_full = evolve_rule(RULE110, width=768, steps=initial_steps, seed=seed110)
grid30_full  = evolve_rule(RULE30,  width=768, steps=initial_steps, seed=seed30)
grid32_full  = evolve_rule(RULE32,  width=768, steps=initial_steps, seed=seed32)
grid184_full = evolve_rule(RULE184, width=768, steps=initial_steps, seed=seed184)
print("初始生成完成")

# 第二步：用Rule 110计算动态高度（并显示所有规则的块熵统计）
print("\n" + "="*70)
print("第二步：活跃区判定（块熵 + 连续低熵块）")
print("="*70)

# ===== 可配置参数区域 =====
BLOCK_HEIGHT = 40        # 块的高度
NGRAM_SIZE = 3           # 内部滑块大小（3表示3×3）
NUM_INTERVALS = 20       # 熵值分区数量
CUTOFF_INTERVAL = 1      # 截断区间（1=最低熵区间，2=前两个低熵区间，依此类推）
CONSECUTIVE_COUNT = 5    # 需要连续多少个低熵块才截断（避免偶然低熵）
# ===========================

print(f"参数配置：")
print(f"  块高度 = {BLOCK_HEIGHT}")
print(f"  内部滑块 = {NGRAM_SIZE}×{NGRAM_SIZE}")
print(f"  熵值分区数 = {NUM_INTERVALS}")
print(f"  截断区间 = 区间{CUTOFF_INTERVAL}（遇到此区间及以下的低熵块就截断）")
print(f"  连续阈值 = {CONSECUTIVE_COUNT}（需要连续{CONSECUTIVE_COUNT}个低熵块才触发截断）")

# 计算所有规则的块熵统计（用于诊断）
def get_entropy_stats(grid, block_height=20):
    """获取所有块的熵值统计"""
    steps, width = grid.shape
    entropy_values = []
    for y in range(0, steps - block_height + 1, block_height):
        block = grid[y:y+block_height, :]
        entropy = calculate_block_entropy(block, block_height, width, NGRAM_SIZE, NGRAM_SIZE)
        entropy_values.append(entropy)
    return np.array(entropy_values)

print("\n各规则的块熵统计：")
for name, grid_full in [("Rule 110", grid110_full), ("Rule 30", grid30_full), 
                         ("Rule 32", grid32_full), ("Rule 184", grid184_full)]:
    entropies = get_entropy_stats(grid_full, block_height=BLOCK_HEIGHT)
    if len(entropies) > 0:
        print(f"  {name:12s}: 均值={np.mean(entropies):.4f}, "
              f"最小={np.min(entropies):.4f}, 最大={np.max(entropies):.4f}, "
              f"标准差={np.std(entropies):.4f}")

dynamic_height = compute_dynamic_height(
    grid110_full, 
    block_height=BLOCK_HEIGHT, 
    ngram_height=NGRAM_SIZE, 
    ngram_width=NGRAM_SIZE, 
    num_intervals=NUM_INTERVALS,
    cutoff_interval=CUTOFF_INTERVAL,
    consecutive_count=CONSECUTIVE_COUNT
)
print(f"\n统一动态高度: {dynamic_height} / {initial_steps}（保留 {dynamic_height/initial_steps*100:.1f}%）")
print(f"说明：截断位置是连续{CONSECUTIVE_COUNT}个区间{CUTOFF_INTERVAL}（低熵/不活跃）块的第一个块之前")

# 显示Rule 110的详细块熵分布（前20个块和截断点附近）
print(f"\nRule 110 块熵详细分析：")
entropies_110 = get_entropy_stats(grid110_full, block_height=BLOCK_HEIGHT)
if len(entropies_110) > 0:
    min_e = np.min(entropies_110)
    max_e = np.max(entropies_110)
    range_e = max_e - min_e
    cutoff_threshold = min_e + CUTOFF_INTERVAL / NUM_INTERVALS * range_e
    
    # 计算每个块所属的区间
    def get_interval(entropy, min_e, range_e, num_intervals):
        if range_e == 0:
            return num_intervals // 2
        normalized = (entropy - min_e) / range_e
        interval = int(normalized * num_intervals) + 1
        if interval > num_intervals:
            interval = num_intervals
        return interval
    
    # 标记低熵块
    def is_low_entropy(entropy, threshold):
        return entropy < threshold
    
    cutoff_pos = dynamic_height // BLOCK_HEIGHT
    print(f"  前10个块的熵值和区间：")
    for i in range(min(10, len(entropies_110))):
        interval = get_interval(entropies_110[i], min_e, range_e, NUM_INTERVALS)
        is_low = is_low_entropy(entropies_110[i], cutoff_threshold)
        low_marker = "[低熵]" if is_low else "      "
        cutoff_marker = " <-- 截断点（连续低熵开始）" if i == cutoff_pos else ""
        print(f"    块{i:3d} (行{i*BLOCK_HEIGHT:4d}-{(i+1)*BLOCK_HEIGHT:4d}): 熵={entropies_110[i]:.4f}, 区间{interval:2d} {low_marker}{cutoff_marker}")
    
    if cutoff_pos > 10 and cutoff_pos < len(entropies_110):
        print(f"  ...")
        print(f"  截断点附近的块（前后各5个）：")
        for i in range(max(0, cutoff_pos-5), min(len(entropies_110), cutoff_pos+CONSECUTIVE_COUNT+2)):
            interval = get_interval(entropies_110[i], min_e, range_e, NUM_INTERVALS)
            is_low = is_low_entropy(entropies_110[i], cutoff_threshold)
            low_marker = "[低熵]" if is_low else "      "
            cutoff_marker = " <-- 截断点（连续低熵开始）" if i == cutoff_pos else ""
            in_consecutive = ""
            if cutoff_pos <= i < cutoff_pos + CONSECUTIVE_COUNT:
                in_consecutive = f" ← 连续低熵块 {i-cutoff_pos+1}/{CONSECUTIVE_COUNT}"
            print(f"    块{i:3d} (行{i*BLOCK_HEIGHT:4d}-{(i+1)*BLOCK_HEIGHT:4d}): 熵={entropies_110[i]:.4f}, 区间{interval:2d} {low_marker}{cutoff_marker}{in_consecutive}")

# 第三步：裁剪所有grids到动态高度
print("\n" + "="*70)
print("第三步：裁剪所有CA图到统一高度")
print("="*70)
grid110 = grid110_full[:dynamic_height, :]
grid30  = grid30_full[:dynamic_height, :]
grid32  = grid32_full[:dynamic_height, :]
grid184 = grid184_full[:dynamic_height, :]
print(f"裁剪完成：所有CA图高度统一为 {dynamic_height} 步")
print("="*70)

def build_features_dilated(grid, horizon, dilation, t_stride=10, x_stride=10, max_samples=1500):
    T, W = grid.shape
    t_max = T - horizon - 1
    if t_max < 1 or W < (2*dilation+1):
        return np.array([]), np.array([]), np.array([])
    times = np.arange(0, t_max+1, t_stride, dtype=np.int32)
    feats_list, labs_list, ts_list = [], [], []
    for t in times:
        row = grid[t]
        L = np.roll(row, dilation)
        C = row
        R = np.roll(row, -dilation)
        feat_full = (L << 2) | (C << 1) | R  # 0..7
        idxs = np.arange(0, W, x_stride, dtype=np.int32)
        feats_list.append(feat_full[idxs])
        labs_list.append(grid[t + horizon, idxs])
        ts_list.append(np.full_like(idxs, t, dtype=np.int32))
    feats = np.concatenate(feats_list)
    labs  = np.concatenate(labs_list)
    ts    = np.concatenate(ts_list)
    if feats.shape[0] > max_samples:
        rng = np.random.default_rng(123)
        sel = rng.choice(feats.shape[0], size=max_samples, replace=False)
        feats = feats[sel]; labs = labs[sel]; ts = ts[sel]
    return feats.astype(np.int32), labs.astype(np.int32), ts.astype(np.int32)

def train_table(features, labels):
    maj = 1 if labels.sum() >= (labels.size - labels.sum()) else 0
    mask1 = labels == 1
    counts1 = np.bincount(features[mask1], minlength=8)
    counts0 = np.bincount(features[~mask1], minlength=8)
    table = np.full(8, maj, dtype=np.int32)
    table[counts1 > counts0] = 1
    table[counts0 > counts1] = 0
    return table, maj

def eval_one_level(grid, t0, Th, x0, Xw, s, h,
                   train_ratio=0.6, t_stride=10, x_stride=10, max_samples=1500):
    sub = grid[t0:t0+Th, x0:x0+Xw]
    feats, labs, ts = build_features_dilated(sub, h, s, t_stride, x_stride, max_samples)
    if feats.size < 80:
        return None  # insufficient
    t_min, t_max = ts.min(), ts.max()
    boundary = t_min + int((t_max - t_min + 1) * train_ratio)
    train_mask = ts <= boundary
    test_mask  = ts >  boundary
    if np.sum(test_mask) < 40 or np.sum(train_mask) < 40:
        return None
    ft, lt = feats[train_mask], labs[train_mask]
    fe, le = feats[test_mask],  labs[test_mask]
    table, maj = train_table(ft, lt)
    pred = table[fe]
    acc  = (pred == le).mean()
    base = (np.full_like(le, maj) == le).mean()
    lift = (acc/(1-acc)) / (base/(1-base)) if 0<acc<1 and 0<base<1 else 1.0
    return lift

def compute_Lk_table(grid, n_regions=240, Th=256, Xw=256, levels=None):
    """
    计算 k层可预测性（L_k）表
    
    对给定 CA，随机采样区域，在 L1–L30 上评估 lift（相对多数基线的胜算比），
    返回包含列 L1…L30 的 DataFrame；无效或 ≤0 记为 NaN。
    
    参数:
        grid: CA时空图
        n_regions: 采样区域数量
        Th, Xw: 区域高度和宽度（固定面积）
        levels: 层级列表，默认为 [1,2,3,...,30]
    
    返回:
        DataFrame，每行一个区域，包含各层级的lift值（无效值用NaN标记）
    """
    if levels is None:
        levels = list(range(1, 31))  # L1 到 L30
    
    rng = np.random.default_rng(424242)
    T, W = grid.shape
    rows = []
    for _ in range(n_regions):
        t0 = int(rng.integers(0, max(1, T - Th + 1)))
        x0 = int(rng.integers(0, max(1, W - Xw + 1)))
        lifts = []
        for level in levels:
            val = eval_one_level(grid, t0, Th, x0, Xw, s=level, h=level,
                                 train_ratio=0.6, t_stride=10, x_stride=10, max_samples=1500)
            # 无效值用 NaN 标记，而不是 1.0
            if val is None or not np.isfinite(val) or val <= 0:
                lifts.append(np.nan)
            else:
                lifts.append(float(val))
        
        rows.append({"t0":t0, "x0":x0, "Th":Th, "Xw":Xw, **{f"L{d}":l for d,l in zip(levels,lifts)}})
    return pd.DataFrame(rows)

print("\n" + "="*70)
print("计算所有层级 L1 到 L30（k层可预测性）")
print("="*70)

df110 = compute_Lk_table(grid110, n_regions=240, Th=256, Xw=256)
df30  = compute_Lk_table(grid30,  n_regions=240, Th=256, Xw=256)
df32  = compute_Lk_table(grid32,  n_regions=240, Th=256, Xw=256)
df184 = compute_Lk_table(grid184, n_regions=240, Th=256, Xw=256)

# ====== 数据验证：各层级统计 ======
print("\n" + "="*70)
print("各层级数据概览")
print("="*70)
for rule_name, df in [("Rule 110", df110), ("Rule 30", df30), ("Rule 32", df32), ("Rule 184", df184)]:
    print(f"\n{rule_name}:")
    level_cols = [col for col in df.columns if col.startswith('L')]
    for col in level_cols[:5]:  # 只显示前5个层级
        vals = df[col].values
        valid_vals = vals[np.isfinite(vals)]
        if len(valid_vals) > 0:
            print(f"  {col}: 有效样本={len(valid_vals):3d}, 均值={np.mean(valid_vals):8.4f}, 范围=[{np.min(valid_vals):7.4f}, {np.max(valid_vals):7.4f}]")
        else:
            print(f"  {col}: 无有效样本")
print("="*70)

def leading_digit_array(x):
    x = np.asarray(x, dtype=float)
    x = x[(x>0)&np.isfinite(x)]
    if x.size == 0: return np.array([], dtype=int)
    m = x.copy()
    while True:
        small = m < 1
        if not np.any(small): break
        m[small] *= 10.0
    while True:
        big = m >= 10
        if not np.any(big): break
        m[big] /= 10.0
    return (m // 1).astype(int)

digits = np.arange(1,10)
benford = np.log10(1 + 1/digits)

def benford_distribution(vals):
    """计算首位数字分布"""
    digs = leading_digit_array(vals)
    counts = np.bincount(digs, minlength=10)[1:10].astype(float)
    emp = counts / counts.sum() if counts.sum()>0 else np.zeros(9)
    return emp

# ====== 保存 CSV 文件 ======
csv110 = "Rule110_Lk_AllLevels.csv"
csv30  = "Rule30_Lk_AllLevels.csv"
csv32  = "Rule32_Lk_AllLevels.csv"
csv184 = "Rule184_Lk_AllLevels.csv"

df110.to_csv(csv110, index=False)
df30.to_csv(csv30, index=False)
df32.to_csv(csv32, index=False)
df184.to_csv(csv184, index=False)

print(f"\n保存CSV文件: {csv110}, {csv30}, {csv32}, {csv184}")

# ====== 计算各层级的统计量（均值和极差）======
print("\n" + "="*70)
print("计算各层级的统计量")
print("="*70)

level_cols = [f"L{i}" for i in range(1, 31)]

def get_level_stats(df):
    """计算每个层级的均值和极差（忽略NaN）"""
    means = []
    ranges = []
    for col in level_cols:
        vals = df[col].values
        valid_vals = vals[np.isfinite(vals)]
        if len(valid_vals) > 0:
            means.append(np.mean(valid_vals))
            ranges.append(np.max(valid_vals) - np.min(valid_vals))
        else:
            means.append(np.nan)
            ranges.append(np.nan)
    return np.array(means), np.array(ranges)

means110, ranges110 = get_level_stats(df110)
means30, ranges30 = get_level_stats(df30)
means32, ranges32 = get_level_stats(df32)
means184, ranges184 = get_level_stats(df184)

# ====== 图1：2×2子图 - 箱式图显示各规则在各尺度的 L_k 分布 ======
print("\n绘制图1：2×2子图 - 箱式图显示各规则在各尺度的 L_k 分布")
fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='#F8F9FA')
x_levels = np.arange(1, 31)

# 准备数据和子图配置 - 按照Class I, II, III, IV顺序
data_list = [
    (df32, 'Rule 32 (Class I)', axes[0, 0], '#95E1D3'),     # 左上：Class I
    (df184, 'Rule 184 (Class II)', axes[0, 1], '#F38181'),  # 右上：Class II
    (df30, 'Rule 30 (Class III)', axes[1, 0], '#4ECDC4'),   # 左下：Class III
    (df110, 'Rule 110 (Class IV)', axes[1, 1], '#FF6B6B'),  # 右下：Class IV
]

for df, title, ax, color in data_list:
    ax.set_facecolor('#F8F9FA')  # 设置子图背景色
    
    # 准备箱式图数据
    data_for_boxplot = []
    for level in range(1, 31):
        col = f"L{level}"
        vals = df[col].values
        valid_vals = vals[np.isfinite(vals)]
        data_for_boxplot.append(valid_vals)
    
    # 绘制箱式图
    bp = ax.boxplot(data_for_boxplot, positions=x_levels, widths=0.6,
                     patch_artist=True,
                     boxprops=dict(facecolor=color, alpha=0.75, linewidth=1.5, edgecolor='#34495E'),
                     medianprops=dict(color='#2C3E50', linewidth=2.5),
                     whiskerprops=dict(color='#7F8C8D', linewidth=1.5),
                     capprops=dict(color='#7F8C8D', linewidth=1.5),
                     flierprops=dict(marker='o', markerfacecolor=color, markersize=4, 
                                   alpha=0.5, markeredgecolor='none'))
    
    # 叠加中位数连线
    medians = []
    for level in range(1, 31):
        col = f"L{level}"
        vals = df[col].values
        valid_vals = vals[np.isfinite(vals)]
        if len(valid_vals) > 0:
            medians.append(np.median(valid_vals))
        else:
            medians.append(np.nan)
    
    ax.plot(x_levels, medians, color='#2C3E50', linewidth=2.5, marker='D', 
            markersize=6, zorder=10, alpha=0.9, 
            markeredgewidth=1, markeredgecolor='white')
    
    ax.set_xlabel('层级 (Lx)', fontsize=11, fontweight='bold', color='#2C3E50')
    ax.set_ylabel('Lift（k层可预测性）', fontsize=11, fontweight='bold', color='#2C3E50')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10, color='#2C3E50')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='#95A5A6')
    # 设置x轴刻度：显示1到30，但标签每隔2个显示一次，避免太密集
    ax.set_xticks(x_levels)
    ax.set_xticklabels([str(i) if i % 3 == 1 else '' for i in x_levels], fontsize=9)
    ax.set_xlim(0, 31)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_color('#7F8C8D')
    ax.spines['bottom'].set_color('#7F8C8D')

fig.suptitle('各规则在各尺度的 L_k 分布（箱式图）', fontsize=16, fontweight='bold', y=0.995, color='#2C3E50')
plt.tight_layout()
plt.savefig('Lk_Boxplots_AllRules.png', dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
print("保存图1: Lk_Boxplots_AllRules.png")

# ====== 找出极差最大的层级（排除L1）======
print("\n" + "="*70)
print("找出极差最大的层级（排除L1）")
print("="*70)

# 排除L1，即从索引1开始（L2到L17）
ranges110_noL1 = ranges110[1:]
ranges184_noL1 = ranges184[1:]

# 找到极差最大的层级
max_idx_110 = np.nanargmax(ranges110_noL1) + 1  # +1因为排除了L1
max_idx_184 = np.nanargmax(ranges184_noL1) + 1

max_level_110 = max_idx_110 + 1  # +1因为层级从1开始
max_level_184 = max_idx_184 + 1

print(f"Rule 110: 极差最大的层级是 L{max_level_110}, 极差 = {ranges110[max_idx_110]:.6f}")
print(f"Rule 184: 极差最大的层级是 L{max_level_184}, 极差 = {ranges184[max_idx_184]:.6f}")

# ====== 图2：频率直方图 - 极差最大层级的本福特分析 ======
print("\n绘制图2：频率直方图 - 本福特分析")

# 提取对应层级的值
col_110 = f"L{max_level_110}"
col_184 = f"L{max_level_184}"

vals_110 = df110[col_110].values
vals_184 = df184[col_184].values

# 过滤有效值
valid_vals_110 = vals_110[np.isfinite(vals_110) & (vals_110 > 0)]
valid_vals_184 = vals_184[np.isfinite(vals_184) & (vals_184 > 0)]

print(f"Rule 110 {col_110}: 有效样本数 = {len(valid_vals_110)}")
print(f"Rule 184 {col_184}: 有效样本数 = {len(valid_vals_184)}")

# 计算首位数字分布
emp110 = benford_distribution(valid_vals_110)
emp184 = benford_distribution(valid_vals_184)

# 计算统计指标
def calculate_benford_stats(observed_dist, n_samples):
    """计算本福特统计指标"""
    digits_range = np.arange(1, 10)
    benford_p = np.log10(1 + 1 / digits_range)
    
    # 观测和期望计数
    obs_counts = observed_dist * n_samples
    exp_counts = benford_p * n_samples
    
    # Chi-square检验
    chi2, p_value = chisquare(f_obs=obs_counts, f_exp=exp_counts)
    
    # MAD (Mean Absolute Deviation)
    mad = np.mean(np.abs(observed_dist - benford_p))
    
    return chi2, p_value, mad

chi2_110, p_110, mad_110 = calculate_benford_stats(emp110, len(valid_vals_110))
chi2_184, p_184, mad_184 = calculate_benford_stats(emp184, len(valid_vals_184))

print(f"\n{'='*70}")
print(f"本福特统计指标")
print(f"{'='*70}")
print(f"Rule 110 - L{max_level_110}:")
print(f"  Chi-square = {chi2_110:.4f}")
print(f"  p-value    = {p_110:.6f}")
print(f"  MAD        = {mad_110:.6f}")
print(f"\nRule 184 - L{max_level_184}:")
print(f"  Chi-square = {chi2_184:.4f}")
print(f"  p-value    = {p_184:.6f}")
print(f"  MAD        = {mad_184:.6f}")
print(f"{'='*70}")

# 绘制直方图 - 使用现代化配色方案
fig, ax = plt.subplots(figsize=(14, 7), facecolor='#F8F9FA')
ax.set_facecolor('#F8F9FA')  # 浅灰背景，更柔和
x = np.arange(1, 10)
width = 0.35

# 使用更现代的配色：柔和的渐变色
color_110 = '#FF6B6B'  # 珊瑚红
color_184 = '#4ECDC4'  # 青绿色
color_benford = '#2C3E50'  # 深蓝灰

# 绘制Rule 110和Rule 184的柱状图
bars1 = ax.bar(x - width/2, emp110, width=width, label=f"Rule 110 - L{max_level_110}", 
               alpha=0.85, color=color_110, edgecolor='white', linewidth=2)
bars2 = ax.bar(x + width/2, emp184, width=width, label=f"Rule 184 - L{max_level_184}", 
               alpha=0.85, color=color_184, edgecolor='white', linewidth=2)

# 本福特定律用虚线标记 - 更粗更明显
ax.plot(x, benford, color=color_benford, linewidth=3.5, linestyle='--', marker='o', 
        markersize=11, label="Benford's Law", zorder=10, markeredgewidth=2.5, 
        markeredgecolor='white', markerfacecolor=color_benford)

ax.set_xlabel('首位数字', fontsize=13, fontweight='bold', color='#2C3E50')
ax.set_ylabel('频率', fontsize=13, fontweight='bold', color='#2C3E50')
ax.set_title(f'首位数字分布对比 (极差最大层级)', fontsize=15, fontweight='bold', pad=15, color='#2C3E50')
ax.legend(fontsize=11, loc='upper right', framealpha=0.98, edgecolor='#BDC3C7', fancybox=True, shadow=True)
ax.set_xticks(x)
ax.set_ylim(0, max(max(emp110), max(emp184), max(benford)) * 1.15)

# 在图上添加统计信息
stats_text = f'Rule 110 - L{max_level_110}:\n  χ² = {chi2_110:.2f}, p = {p_110:.4f}, MAD = {mad_110:.4f}\n\n'
stats_text += f'Rule 184 - L{max_level_184}:\n  χ² = {chi2_184:.2f}, p = {p_184:.4f}, MAD = {mad_184:.4f}'
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#BDC3C7'),
        family='monospace')

# 网格样式改进 - 更细更淡
ax.grid(True, alpha=0.25, axis='y', linestyle='--', linewidth=0.8, color='#95A5A6')
ax.set_axisbelow(True)

# 去掉上边框和右边框，更现代
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_color('#7F8C8D')
ax.spines['bottom'].set_color('#7F8C8D')

plt.tight_layout()
plt.savefig('Benford_PeakLevels.png', dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
print("保存图2: Benford_PeakLevels.png")

plt.show()

print("\n" + "="*70)
print("分析完成！")
print("="*70)
