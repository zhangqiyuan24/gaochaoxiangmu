# 水下气体泄漏被动声学定位 —— 水池实验数据分析管线

实验室规模被动声学定位研究的完整分析代码与结果：8 元平面环形水听器阵
（间距 0.5 m）对水下气泡 / 换能器声源做 **二维近场定位**，对比四种波束形成定位器：

- **CBF** —— 常规（延时叠加）波束形成
- **MVDR** —— 自适应波束形成（对角加载 Capon）
- **MUSIC** —— 基于原始外积矩阵的子空间方法
- **CSDM-MUSIC** —— 基于互谱密度矩阵（CSDM）的改进 MUSIC

定位方式：球面波（近场）导向模型 + 二维位置网格直接搜索，网格 1/36 m，
x ∈ [−3, 3] m、y ∈ [0, 6] m；使用单快拍谱的 23 个分析频点（1000:500:12000 Hz）。

---

## 一、目录结构

```
code/                     全部脚本（按管线顺序）
  beamforming_port.py     ★ 核心：四种定位器的 numpy 复刻（含 MATLAB 原始"怪癖"：
                          非共轭/共轭两种矩阵形式、1/8 缩放后加 δ=1e-3、
                          MUSIC 图上下翻转、逐行先到先取 argmax）
  val_hist8.py            阵列重装定标：逐条件扫描比例因子 s（约 1.20–1.26），
                          与归档逐事件误差表对齐
  run_315_tone.py         E1（换能器音调，21 条件 × 4 算法 × ≤20 事件）：
                          逐事件误差 CSV + 事件平均功率图口径复核
  run_315_full.py         E1 旧版全量跑（含气泡工况），同时导出代表性空间谱
                          .npz 图（Fig.2 素材）
  avg_maps.py             事件平均功率图 vs 《误差真值表》逐格验证
  fig2.py                 Fig.2 草稿渲染（matplotlib 版）
  run_e2_snr.py           E2：SNR 扫描（后处理注噪，见下文设计说明）
  fig4_snr.mjs            Fig.4 生成器：误差–SNR 曲线（纯 pgfplots，Node.js）
  dump_315_tables.py      归档 xlsx 误差表批量导出 CSV（数据溯源用）
  dump_xlsx.py            归档 xlsx / wav 头信息检查（数据溯源用）

results/                  各次运行的输出（可直接查阅，不依赖原始数据）
  val_hist8.*             定标日志与逐条件 s 值
  e1_errors.csv           E1 逐事件误差（列：algorithm, signal_type, distance_m,
                          sample_id, error_m, est_x_m, est_y_m, freq_band_Hz）
  e1_v3_report.json       E1 逐条件统计 + 与归档/真值表对照
  e2_snr.csv              E2 汇总（条件×算法×SNR：均值/标准差/中位数/成功率）
  e2_snr_events.csv       E2 逐事件明细
  e2_summary.json         E2 摘要（含各算法临界 SNR）
  fig2/                   Fig.2（平均功率图）全套：成品 PDF/PNG、
                          面板 PNG + markers.json、可独立编译的 tex 源
```

## 二、环境要求

- **Python 3.8+**：`numpy`、`matplotlib`（fig2 草稿）；`openpyxl` / `xlrd`
  仅在跑数据溯源脚本（dump_*.py）时需要
- **Node.js**（任意较新版本）：仅 Fig.4 生成器需要
- Fig.2 / Fig.4 的 tex 编译用 LaTeX（推荐 tectonic），只依赖 `pgfplots`

## 三、原始数据（不含在仓库内）

原始 8 通道 wav（24-bit PCM、64 kHz、10 ms 突发）为水池实验私有数据，**不在仓库内**。
脚本按如下目录树读取（复现 E1/E2 需要把 wav 放回这个结构）：

```
<data_root>/
  bubble wave-2024.3.15 1/
    1000Hz 1/ 1000Hz 2/ ... 13000Hz 3/     # 换能器音调条件：<频率>Hz <距离>/
    *.wav
```

注意：事件顺序必须与 MATLAB `dir()` 一致（按含扩展名的文件名排序，
即 `bubble1.wav < bubble10.wav < bubble2.wav`），管线已内建该排序。
源战役中 5 kHz/2 m、7 kHz/2 m、7 kHz/3 m 三个条件各只有 1 个有效录音、
13 kHz/3 m 为空目录，脚本会自动处理。

**没有原始数据也能做的事**：查阅 results/ 全部结果、重新编译 Fig.2 tex、
用 `results/e2_snr.csv` 重画 Fig.4。

## 四、复现步骤（有原始数据时）

```bash
# 1) 定标（~1.5 h）：逐条件确定阵列重装比例因子 s
python3 code/val_hist8.py

# 2) E1 全量重跑（换能器音调，21 条件）
python3 code/run_315_tone.py

# 3) E2 SNR 扫描（三个面板条件；可自行增删）
E2_CONDS=5000Hz_3,3000Hz_2,11000Hz_1 python3 code/run_e2_snr.py

# 4) Fig.4
node code/fig4_snr.mjs results/e2_snr.csv <输出目录> 5000Hz_3,3000Hz_2,11000Hz_1 4.5
```

脚本里的 `~/vsip/...` 路径按需改成自己的数据/输出位置。

## 五、方法口径（重要，防误读）

1. **两种误差口径**：表格/真值表用的是"事件平均功率图的峰值位置误差"
   （先把 N 个事件的功率图平均、再找峰）；逐事件 CSV 是每个事件各自的误差。
   重复音调脉冲是确定性副本时两者会重合（如 5 kHz/3 m 的 CBF/MVDR）。
2. **E2 注噪设计**：在零填充单快拍谱的 23 个分析频点上直接叠加共轭对称
   复高斯（等价于实带限线谱噪声），SNR 定义在阵列平均频点功率上；
   同一（条件,事件,SNR）的四个算法使用**同一份噪声实现**（随机种子可复现）。
   这是**后处理注入**，不是水池实测噪声。
3. **MUSIC 族的可复现性**：单快拍秩-1 矩阵的噪声子空间在简并特征簇内
   基任意，MATLAB `eig` 与 LAPACK 返回不同（都合法）的正交基，因此投影子
   依赖基选择——MUSIC/CSDM-MUSIC 只能统计口径复现（均值/中位数吻合，
   逐事件可能不同），这是线性代数事实而非实现错误。

## 六、与论文结果的对应与校验状态

| 论文内容 | 仓库来源 | 校验状态 |
|---|---|---|
| Table I/III 数字 | 归档 xlsx（真值表） | 仓库结果与之对照 |
| 5 kHz/3 m CBF 0.639 / MVDR 0.194 m | `results/e1_v3_report.json` | **逐事件精确复现（20/20）** |
| 7 kHz/1 m、9 kHz/1 m CBF | 同上 | **逐事件精确复现（20/20）** |
| Fig.2（平均功率图） | `results/fig2/` | 面板标记距离 = 真值表逐格相等 |
| Fig.4 / IV-C（临界 SNR +10/+5/+10 dB） | `results/e2_snr.csv` | 曲线与论文内联图逐点一致 |

## 七、许可

代码 MIT（见 LICENSE）；结果 CSV/JSON/PNG 供审稿复现使用。
