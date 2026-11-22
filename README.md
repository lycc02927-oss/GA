# 六种 ED 工况的参数寻优

本仓库按 ED 模型分别选择参数，复用 `lycc02927-oss/test` 的三维单发量测模型、LHS、GPR 和 EI。量测目标为 ED。最大值模式用排名轮盘赌遗传算法选择十个参数。平台区模式只有一个生效参数，单独遍历六个阈值。

- [六种工况、参数对照及实测分析](REPORT.md)
- [每个模型的入选参数](results/selected.csv)
- [独立种子验证汇总](results/validation.csv)
- [逐次实验、最佳位置、停止原因及得分](results/trials.csv)
- [各代候选及排名](results/generations.csv)
- [各模型的无抖动数值参考](results/references.csv)

## 安装和复现

本次使用 Python 3.12；下列锁定版本建议搭配 Python 3.12，依赖 NumPy、SciPy、Matplotlib。以下命令假设两个仓库检出到同一父目录。版本号和随机种子记录在 [实验配置](results/config.json)。复现时优先使用记录的版本。

```bash
git clone https://github.com/lycc02927-oss/GA.git
git clone https://github.com/lycc02927-oss/test.git
git -C test checkout 32f0d18437fbc73a0c1458bb121286c59f4ed2b6
cd GA
python -m pip install numpy==2.3.5 scipy==1.17.0 matplotlib
python tune_ed.py --source ../test --workers 8 --output my_run
```

复现多峰专用实验的完整指令如下。

```bash
python tune_ed.py --source ../test --profile multi_peak --models 6 --population 6 --generations 3 --train-seeds 60 61 62 63 --valid-seeds 342 343 344 345 346 347 348 349 350 351 --initial 40 --rounds 45 --workers 8 --output multipeak_run
```

使用新的输出目录，避免历史结果被覆盖。模型编号 1–6 依次为 P3 试验、LT 仿真、LT 精细仿真、中心高斯、偏置高斯、多峰高斯。参数候选集中定义在 `tune_ed.py` 的 `space()` 和 `PLATEAU` 中。训练和验证种子各自唯一且互不重叠；同一阶段的各候选、基线使用同一批种子。

## 算法与结果用法

1. `tune_ed.py` 为每个模型建立独立种群，默认 8 个个体、4 代，初代含默认窗口对照。一个个体包含十个 Maximum 参数。同一候选在本模型中只做一次训练评估。
2. 先按落入数值参考峰值 10% 范围内的试验次数排名，再比较 `平均误差百分点 + 0.12 × 平均总量测次数`，越小越好。父代按排名获得由高到低的线性轮盘权重，冠军进入下一代，其余个体逐位交叉并以 18% 概率重抽基因。
3. 用全部已评估个体的训练成绩选一组最大值模式参数。平台区阈值单独评分。十个新种子只用于评估选中的方案和基线，不参与选择。
4. 在 `selected.csv` 找到模型行。最大值模式使用从 `gp_restarts` 到 `min_dis` 的十列；平台区模式使用同一行的 `platform_error`。要原样重放某一组，使用下方接口。部署到设备前先核对参数的接收层和坐标尺度，特别是 `gp_restarts` 和 `min_dis`。
5. `trials.csv` 中 `observed_ed` 是带抖动的单发读数。Maximum 下 `x,y,z` 是历史最高读数的设定坐标，`scoring_ed` 是此坐标无抖动 ED。Plateau 下 `x,y,z` 是平台算法返回的中心，`scoring_ed` 是中心及轴向邻域七个探针中的最低无抖动 ED。`reference_ed` 为同一模型的无抖动数值参考峰值；七点评分不代表平台宽度的实测真值。
6. `bo_rounds` 只计已量测的 BO 候选；`evaluations` 包含初始点。停止原因 `window` 为近期波动达到阈值，`short_distance` 为连续三个候选过近，`budget` 为达到预算，`no_improve` 为上游耐心计数到限。平台的 `axes_k` 表示初始门槛通过的轴数。本地试验没有设备回位复测或 RPC。
7. `validation.csv` 的 `GA/sweep` 是训练入选配置，`baseline` 是同预算上游默认最大值配置或平台阈值 0.02。`window_baseline` 在 Maximum 原生配置上开启 5 点 / 10% 窗口和 0.5 最近点阈值。`hits/runs` 与平均误差、总量测数应一起读。更换目标、坐标范围、量测模型或预算后，应重新调参和复验。

### 只重放选中的组合

以下命令从选定目录读取该模型的参数、预算及参考峰，直接输出每次寻优的设定位置、单发 ED、评分、BO 轮数、总量测次数与停止原因。无需重新训练 GA。

```bash
python replay.py --source ../test --run results --model 4 --seeds 242 243
python replay.py --source ../test --run results --model 5 --mode plateau --seeds 242
python replay.py --source ../test --run multipeak --model 6 --seeds 342 343
```

标准预算和多峰专用预算分别存放在 [results](results) 与 [multipeak](multipeak)。两组验证使用不同种子；[标准组在多峰新种子上的复验](multipeak/standard_replay.csv) 用于同种子比较。
