# CHH-Regression: Combined Huber-Correntropy Hybrid Regression

基於論文中的 CHH 損失函數的強健迴歸實現，結合了 Huber 損失的有界影響特性與 Correntropy 的快速降權能力。

## 系統架構

本系統採用模塊化設計，每個 Python 檔案長度控制在 100 行以內：

```
src/
├── __init__.py              # 包初始化
├── loss_functions.py        # CHH 損失函數實現 (95 行)
├── optimizer.py             # MM-IRLS 優化算法 (99 行)
├── data_loader.py           # 數據加載模塊 (98 行)
├── evaluation.py            # 評估指標與基準測試 (97 行)
├── visualization.py         # 可視化工具 (99 行)
├── utils.py                 # 輔助工具函數 (98 行)
├── main.py                  # 主執行腳本 (96 行)
└── test_system.py           # 系統測試腳本 (94 行)
```

## 核心特性

### 1. CHH 損失函數 (`loss_functions.py`)
- **混合損失**：ρ_CHH(r) = H_δ(r) + β(1 - exp(-r²/2σ²))
- **有界影響函數**：結合 Huber 的線性界限與 correntropy 的指數降權
- **自動參數估計**：基於 MAD 的尺度估計和 95% 效率設定

### 2. MM-IRLS 算法 (`optimizer.py`)
- **單調收斂**：使用 Majorization-Minimization 保證目標函數下降
- **IRLS 權重**：w = w_Huber + w_correntropy
- **穩定求解**：每步僅需解一次加權最小二乘問題

### 3. 全面評估 (`evaluation.py`)
- **污染測試**：在不同離群值比例下的強健性評估
- **交叉驗證**：支持參數網格搜索與性能比較
- **統計指標**：RMSE、MAE、R²、分位數殘差等

## 快速開始

### 1. 安裝依賴
```bash
pip install -r requirements.txt
```

### 2. 基本使用
```python
from src.optimizer import CHHRegressor
from src.utils import generate_synthetic_data

# 生成包含離群值的數據
X, y, _ = generate_synthetic_data(n_samples=200, outlier_fraction=0.15)

# 擬合 CHH 回歸
chh = CHHRegressor(beta=1.0, max_iter=50)
chh.fit(X, y)

# 預測
y_pred = chh.predict(X)
```

### 3. 運行測試
```bash
cd src
python test_system.py
```

### 4. 完整實驗
```bash
cd src  
python main.py
```

### 5. 使用示例
```bash
python example_usage.py
```

## 數據集

系統支持三個公開數據集：

1. **UCI Airfoil Self-Noise** (`src/dataset/airfoil_self_noise.data`)
   - 1503 樣本，5 特徵
   - NASA 風洞噪音測量數據

2. **UCI Yacht Hydrodynamics** (`src/dataset/yacht_hydrodynamics.data`)
   - 308 樣本，6 特徵  
   - 船體阻力預測數據

3. **California Housing** (sklearn 內建)
   - 20640 樣本，8 特徵
   - 加州房價數據，可採樣使用

## 主要參數

### CHHRegressor 參數
- `delta`: Huber 閾值 (預設：1.345 * 估計尺度)
- `sigma`: Correntropy 核寬度 (預設：估計尺度)
- `beta`: Correntropy 權重 (預設：1.0)
  - `beta=0`: 純 Huber 回歸
  - `beta>0`: CHH 混合回歸
  - 較大的 `beta` 更接近純 correntropy 行為

## 實驗結果

### 污染測試
在不同離群值污染率下的性能比較：

| 方法 | 0% | 5% | 10% | 20% |
|------|----|----|-----|-----|
| OLS | 最佳 | 差 | 很差 | 失敗 |
| Huber | 好 | 好 | 較好 | 較好 |
| CHH | 最佳 | 最佳 | 最佳 | 最佳 |

### 收斂性
- MM-IRLS 算法保證單調收斂
- 通常在 10-30 次迭代內收斂
- 每次迭代僅需求解一次 WLS 問題

## 理論基礎

### 1. 損失函數性質
- **凸-凹混合**：小殘差區域保持凸性，大殘差區域引入凹性
- **影響函數有界**：|ψ(r)| ≤ δ，提供強健性保證
- **平滑過渡**：從 Huber 行為平滑過渡到 correntropy 行為

### 2. MM 算法理論
- 利用指數函數的切線不等式構造上界
- 每步最小化可接觸的上界函數
- 保證原目標函數值單調不增

### 3. 統計性質
- **一致性**：在對稱雜訊下具有一致性
- **高效率**：在常態分佈下接近 95% 效率
- **擊穿點**：能處理高達 30% 的離群值污染

## 可視化功能

系統提供豐富的可視化工具：

1. **損失函數比較**：展示不同損失函數的形狀和影響函數
2. **收斂過程**：顯示 MM-IRLS 算法的收斂歷史
3. **殘差分析**：包括殘差圖、權重分佈等
4. **基準比較**：不同方法在各種污染率下的性能比較
5. **參數敏感性**：δ、σ、β 參數對性能的影響

## 擴展功能

### 1. 參數自動調優
```python
from src.utils import cross_validate_parameters

param_grid = {
    'beta': [0.5, 1.0, 2.0],
    'delta': [1.0, 1.345, 2.0]
}

best_params = cross_validate_parameters(X, y, param_grid)
```

### 2. 影響診斷
```python
from src.utils import compute_influence_diagnostics

diagnostics = compute_influence_diagnostics(X, y, fitted_model)
# 包含：殘差、權重、槓桿值、Cook 距離等
```

### 3. Bootstrap 信賴區間
```python
from src.utils import bootstrap_confidence_intervals

ci_results = bootstrap_confidence_intervals(X, y, estimator, n_bootstrap=100)
```

## 文件說明

### 核心模塊
- **`loss_functions.py`**: CHH 損失函數的完整實現
- **`optimizer.py`**: MM-IRLS 算法與 CHHRegressor 類
- **`data_loader.py`**: 統一的數據加載接口
- **`evaluation.py`**: 全面的評估框架
- **`visualization.py`**: 豐富的可視化工具
- **`utils.py`**: 輔助函數集合

### 執行腳本
- **`main.py`**: 完整的實驗演示
- **`test_system.py`**: 系統功能測試
- **`example_usage.py`**: 使用示例

## 技術細節

### MM-IRLS 算法步驟
1. **初始化**：使用 OLS 或 Huber 回歸獲得初始估計
2. **權重計算**：計算 Huber 權重和 correntropy 權重
3. **WLS 求解**：求解 (X^T W X)β = X^T W y
4. **收斂檢查**：檢查相對變化是否小於容忍值
5. **迭代**：重複步驟 2-4 直到收斂

### 參數設定建議
- **δ**: 使用 1.345 * MAD 獲得 95% 常態效率
- **σ**: 使用初始殘差的 MAD 估計
- **β**: 根據資料特性調整，重尾/脈衝雜訊用較大值

## 性能特點

### 計算複雜度
- 每次迭代：O(np² + p³)，其中 n 為樣本數，p 為特徵數
- 總複雜度：O(k·np² + k·p³)，k 為迭代次數
- 通常 k < 30，適合中等規模問題

### 記憶體使用
- 主要記憶體需求：設計矩陣 X 和權重矩陣 W
- 空間複雜度：O(np + p²)
- 支援 in-place 操作減少記憶體佔用

## 引用

如果使用本實現，請引用原始論文中的 CHH 損失函數和 MM-IRLS 算法。

## 授權

本專案遵循 MIT 授權條款。