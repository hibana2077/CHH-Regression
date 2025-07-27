# CHH-Regression：結合 Huber 與 Correntropy 的凸–凹混合強健迴歸與其 MM-IRLS 演算法

---

## 研究核心問題

在含有重尾雜訊與少量極端離群值的線性迴歸中，最小平方法（MSE）易被污染；Huber 損失能界定影響函數且在常態下具高效率，但對「極端」離群值並不會完全降權；相對地，以高斯核為基底的 **correntropy / Welsch** 類損失對大殘差會快速降權、對脈衝雜訊特別穩健。如何**同時**保留 Huber 在小殘差區的穩定凸性與常態效率、又具備 correntropy 在大殘差區的強降權能力，是本研究要解決的核心問題。([維基百科][1], [cran.r-project.org][2], [jmlr.org][3], [arXiv][4])

---

## 研究目標

1. **提出新損失函數（CHH 損失）**，在小殘差時保持 Huber 的二次曲率與 95% 常態效率設定，在大殘差時引入 correntropy 的快速降權，以得到**有界影響函數**且對重尾/脈衝雜訊更穩健。([維基百科][1], [cran.r-project.org][2], [arXiv][4])
2. **給出可證明單調收斂的 MM-IRLS 最佳化演算法**，每步僅解一次加權最小平方法（WLS），CPU 即可快速完成。([people.maths.bris.ac.uk][5], [維基百科][6], [palomar.home.ece.ust.hk][7])
3. **建立統計性質**：影響函數有界；在對稱雜訊下的一致性；並提供與 Huber、Welsch/MCC 的聯繫與極限情形。([維基百科][1], [cran.r-project.org][2], [jmlr.org][3], [projecteuclid.org][8])
4. **以公開迴歸資料集做實證**，比較 OLS、Huber、Welsch/MCC 與所提 CHH 在 MAE/RMSE 與離群污染實驗的表現。([archive.ics.uci.edu][9], [archive.ics.uci.edu][10], [scikit-learn.org][11])

---

## 方法：CHH 損失與可證明可行的 MM-IRLS

### 1) CHH 損失定義

對殘差 $r=y-\mathbf x^\top \boldsymbol\beta$，定義

$$
\rho_{\text{CHH}}(r;\delta,\sigma,\beta)
\;=\;
H_\delta(r)\;+\;\beta\Big(1-\exp\big(-\tfrac{r^2}{2\sigma^2}\big)\Big),
$$

其中 $H_\delta(r)$ 為 Huber 損失；第二項為 **correntropy/Welsch** 誘導之損失（Gaussian ρ），$\beta\!\ge\!0$ 控制其權重，$\sigma\!>\!0$ 為核寬度。當 $\beta\!=\!0$ 時退化為 Huber；當 $\delta\!\to\!\infty$ 時趨近純 correntropy。此式把 Huber 的凸性與 Welsch 的「再下降」降權特性結合在一起。([維基百科][1], [cran.r-project.org][2], [jmlr.org][3])

**性質（要點）：**

* **影響函數有界。** $\psi(r)=\rho'(r)=\psi_{\text{Huber}}(r)+\beta\frac{r}{\sigma^2}e^{-r^2/(2\sigma^2)}$。當 $|r|\!\to\!\infty$，第二項趨 0，$\psi(r)\!\to\!\delta\,\mathrm{sign}(r)$，故有界。([維基百科][1], [cran.r-project.org][2], [projecteuclid.org][8])
* **小殘差高效率。** 在 $|r|\!\ll\!\delta,\sigma$ 時，$\rho_{\text{CHH}}\approx \tfrac{1}{2}r^2 + \beta \tfrac{r^2}{2\sigma^2}$，維持二次曲率，有利於常態情境下的效率；$\delta$ 取 1.345$\hat\sigma$ 可達約 95% 常態效率。([維基百科][1], [經濟與商業科學年刊][12], [SpringerLink][13])
* **大殘差快速降權。** Welsch/correntropy 的 $e^{-r^2/(2\sigma^2)}$ 使權重隨 $|r|$ 指數式衰減，對脈衝型離群特別穩健。([cran.r-project.org][2], [arXiv][4])

### 2) MM 上界與 IRLS 權重

令 $x=\tfrac{r^2}{2\sigma^2}$。利用指數函數的切線不等式

$$
e^{-x}\;\ge\;e^{-x_k}\big(1-(x-x_k)\big)
\;\Rightarrow\;
1-e^{-x}\;\le\; \text{const} + e^{-x_k}\,x,
$$

可得一個在目前點 $x_k$ 之上的**可接觸上界**。因此在第 $k$ 次迭代，整體目標的可解上界為

$$
\sum_i
\underbrace{H_\delta(r_i)}_{\text{Huber 可半二次化}}
\;+\;
\underbrace{\frac{\beta}{2\sigma^2}\,e^{-x_{i,k}}\, r_i^2}_{\text{加權二次項}}
\;+\;\text{const},
$$

這導致一次 **WLS 子問題**。對 Huber，可用半二次（Half-Quadratic）/IRLS 表示，給出權重

$$
w_i^{\text{Huber}}=
\begin{cases}
1, & |r_i|\le \delta,\\[2pt]
\delta/|r_i|, & |r_i|>\delta,
\end{cases}
\qquad
w_{i,k}^{\text{C}}=\dfrac{\beta}{2\sigma^2}\,e^{-\,r_{i,k}^2/(2\sigma^2)}.
$$

令 $w_{i,k}=w_i^{\text{Huber}}+w_{i,k}^{\text{C}}$，解 $(X^\top W_k X)\beta_{k+1}=X^\top W_k y$。由於每步最小化的是對原目標的全域上界且在目前點相切，**目標值單調不增**（MM 性質）。([palomar.home.ece.ust.hk][7], [維基百科][6], [people.maths.bris.ac.uk][5], [BCMI][14], [iSEE][15])

> **可行性說明。** MCC/ Welsch 本身已有 IRLS / 半二次或 MM 推導；此處把 **Huber 的 IRLS 權重**與由上界導出的 **correntropy 指數權重**串接到同一個 WLS 子問題中，滿足 MM 的「上界且相切」兩條件，因此在實作上能穩定下降並於 CPU 上快速收斂。這與文獻中對 correntropy、MM/CCCP、IRLS 的已知結果相容。([palomar.home.ece.ust.hk][7], [維基百科][6], [people.maths.bris.ac.uk][5], [jmlr.org][3], [arXiv][4])

### 3) 參數設定（建議的預設且可交叉驗證）

* **尺度估計**：以 **MAD** 估計殘差尺度，$\hat\sigma = 1.4826\,\mathrm{MAD}$。([維基百科][16], [search.r-project.org][17])
* **Huber 閥值**：$\delta = 1.345\,\hat\sigma$（常態下約 95% 效率）；可交叉驗證微調。([經濟與商業科學年刊][12], [SpringerLink][13], [cran.r-project.org][2])
* **核寬度 $\sigma$**：以初始殘差的 $\hat\sigma$ 或其倍數（如 $0.8\hat\sigma,1.0\hat\sigma,1.2\hat\sigma$）網格搜尋。MCC 文獻指出帶寬會影響降權強度。([jmlr.org][3], [arXiv][4])
* **$\beta$**：控制大殘差的再下降力度，可在 $\{0.2,0.5,1,2\}$ 交叉驗證。當 $\beta\to0$ 時回到 Huber；$\beta$ 大時更接近 Welsch 行為。([cran.r-project.org][2], [jmlr.org][3])

---

## 演算法（CHH–MM–IRLS）

1. **初始化** $\beta^{(0)}$：用 OLS 或 Huber 迴歸。$\sigma,\delta$ 依上節設定。計算殘差 $r^{(0)}$。([維基百科][1])
2. **迭代** $k=0,1,\dots$：

   * 計算 $w_i^{\text{Huber}}$ 與 $w_{i,k}^{\text{C}}=\dfrac{\beta}{2\sigma^2}\exp\!\big(-r_{i,k}^2/(2\sigma^2)\big)$。([維基百科][1], [cran.r-project.org][2])
   * 設 $W_k=\mathrm{diag}(w_i^{\text{Huber}}+w_{i,k}^{\text{C}})$，解 WLS：$(X^\top W_k X)\beta^{(k+1)}=X^\top W_k y$。([people.maths.bris.ac.uk][5])
   * 若相對目標值下降 < $10^{-6}$ 或達上限迭代（如 100）則停止。MM 保證單調不增。([維基百科][6], [palomar.home.ece.ust.hk][7])
3. **輸出** $\hat\beta$ 與殘差診斷（影響度、權重分佈）。影響函數界限由 $|\psi(r)|\le \delta$ 得到。([維基百科][1], [projecteuclid.org][8])

---

## 統計與理論要點（可在論文中證明/呈現）

* **有界影響函數**：由上式可直接計算 $\sup_r |\psi(r)|=\delta$。提供圖示與與 Huber、Welsch 對照。([維基百科][1], [cran.r-project.org][2], [projecteuclid.org][8])
* **DC 分解與 CCCP 視角（可作為備用證明）**：
  $1-e^{-x}=x-\phi(x)$，其中 $\phi(x)=e^{-x}-1+x$ 為凸函數；故 $\rho_{\text{CHH}}=(H_\delta(r)+\beta x)-\beta\phi(x)$ 為凸–凸差（DC）形式，可用 CCCP 線性化凹部份得下降迭代；與上述 MM 上界法一致。([維基百科][6], [palomar.home.ece.ust.hk][7])
* **一致性**：在對稱、以 0 為中心的雜訊下，$\mathbb E[\psi(r)]=0$ 的唯一根在真值附近，與經典 M‑估計的一致性條件一致；可引用 M‑估計框架與影響函數文獻作論述。([projecteuclid.org][8])
* **與既有方法的關係**：$\beta=0$ 回到 Huber；$\delta\to\infty$ 時成為純 correntropy（Welsch）；因此 CHH 連續地涵蓋兩者。([維基百科][1], [cran.r-project.org][2], [jmlr.org][3])

---

## 資料集（公開、可直接下載）

1. **UCI Airfoil Self-Noise**：$n=1503$、5 特徵、無遺漏值；NASA 風洞量測，用於預測聲壓級（迴歸）。適合做離群污染與重尾雜訊測試。([archive.ics.uci.edu][9], [kaggle.com][18], [Medium][19])
2. **UCI Yacht Hydrodynamics**：$n=308$、6 特徵；預測船體阻力。小樣本，便於 CPU 快速交叉驗證。([archive.ics.uci.edu][10], [KXY Technologies][20])
3. **California Housing（scikit‑learn 內建）**：$n=20640$、8 特徵；可隨機抽樣 5–10% 作為大型但可控的附加測試。([scikit-learn.org][11], [inria.github.io][21], [CodeSignal][22])

---

## 實驗設計與評估

* **度量**：RMSE、MAE；另回報加權殘差的中位數與 90/95 分位作穩健性觀察。（RMSE/MAE 為常見做法；穩健文獻常檢查尾部表現與權重。）([jmlr.org][3], [arXiv][4])
* **污染測試**：在訓練資料中以比例 $p\in\{0,5,10,20\%\}$ 隨機挑選樣本，將目標值加入大幅度擾動（如 $\pm 8\hat\sigma$）。比較 OLS、Huber、Welsch/MCC、CHH 的泛化誤差。MCC 在脈衝雜訊下通常優於 MSE/Huber，預期 CHH 會在多數情形與 Welsch/MCC 相當或更佳，且在常態/近常態時不輸 Huber。([arXiv][4], [jmlr.org][3])
* **消融實驗**：固定 $\sigma$ 與 $\delta$，掃描 $\beta$；觀察從 Huber（$\beta=0$）過渡到 Welsch 行為時的性能曲線。([cran.r-project.org][2], [jmlr.org][3])
* **收斂性與時間**：回報每回合目标值（原始 $\sum\rho_{\text{CHH}}$）下降曲線與迭代次數；MM 理論保證單調不增。([維基百科][6], [palomar.home.ece.ust.hk][7])

---

## 預期貢獻與創新

1. **新的損失族 CHH（Huber + correntropy）**：同時具備 Huber 的有界影響與常態效率，以及 correntropy 的再下降降權；以簡單三參數 $(\delta,\sigma,\beta)$ 平滑連續地貫通兩類經典強健損失。此具體型式與其 **MM 可解上界**、**IRLS 權重公式**的推導，為本文主要數學創新與可行性保證。([維基百科][1], [cran.r-project.org][2], [palomar.home.ece.ust.hk][7], [維基百科][6])
2. **單調收斂的 MM–IRLS 演算法**：以切線上界 $e^{-x}$ 的不等式構造 surrogate，將混合損失化為一次 WLS 子問題，證明每步下降。這比直接處理非凸項更安全，且易於以現成線性代數庫在 CPU 上落地。([palomar.home.ece.ust.hk][7], [維基百科][6], [people.maths.bris.ac.uk][5])
3. **實證展示在不同污染率下的穩健優勢**：在 UCI 與 sklearn 公開資料上，系統性比較 OLS/Huber/Welsch(MCC)/CHH，對社群提供一個小而完整、可複現的強健迴歸基準。([archive.ics.uci.edu][9], [archive.ics.uci.edu][10], [scikit-learn.org][11])

---

## 參考文獻（精選）

* Huber 損失與 1.345 調參、IRLS：Huber (1964, 1981) 的概念性介紹可見百科條與後續教科書；95% 效率常用 $1.345$ 與 MAD 常數 1.4826 的來源見下列文獻與套件文件。([維基百科][1], [經濟與商業科學年刊][12], [SpringerLink][13], [cran.r-project.org][2], [維基百科][16], [search.r-project.org][17])
* Correntropy / Welsch、MCC 理論與穩健性：Principe 的 ITL 專書與多篇 MCC 研究、JMLR 的 CIL 理論分析。([Google 書籍][23], [SciSpace][24], [jmlr.org][3], [arXiv][4], [cran.r-project.org][2])
* MM / 半二次 / IRLS / CCCP：經典 MM 維基與教學講義、IRLS 經典論文、半二次文獻。([維基百科][6], [palomar.home.ece.ust.hk][7], [people.maths.bris.ac.uk][5], [BCMI][14], [iSEE][15])
* 影響函數與 M‑估計統計性質：影響函數教科書級論述與近年濃度分析。([projecteuclid.org][8], [arXiv][25])
* 資料集：UCI Airfoil、UCI Yacht、sklearn California Housing。([archive.ics.uci.edu][9], [archive.ics.uci.edu][10], [scikit-learn.org][11], [kaggle.com][18], [inria.github.io][21], [KXY Technologies][20], [CodeSignal][22], [Medium][19])

---

[1]: https://en.wikipedia.org/wiki/Huber_loss?utm_source=chatgpt.com "Huber loss - Wikipedia"
[2]: https://cran.r-project.org/web/packages/robustbase/vignettes/psi_functions.pdf?utm_source=chatgpt.com "[PDF] Definitions of ψ-Functions Available in Robustbase"
[3]: https://www.jmlr.org/papers/volume16/feng15a/feng15a.pdf?utm_source=chatgpt.com "[PDF] Learning with the Maximum Correntropy Criterion Induced Losses ..."
[4]: https://arxiv.org/pdf/1703.08065?utm_source=chatgpt.com "[PDF] Robustness of Maximum Correntropy Estimation Against Large ..."
[5]: https://people.maths.bris.ac.uk/~mapjg/papers/IRLS.pdf?utm_source=chatgpt.com "[PDF] Iteratively Reweighted Least Squares for Maximum Likelihood ..."
[6]: https://en.wikipedia.org/wiki/MM_algorithm?utm_source=chatgpt.com "MM algorithm - Wikipedia"
[7]: https://palomar.home.ece.ust.hk/ELEC5470_lectures/slides_algorithms_MM.pdf?utm_source=chatgpt.com "[PDF] Algorithms: Majorization-Minimization (MM) - HKUST"
[8]: https://projecteuclid.org/journals/bernoulli/volume-23/issue-4B/Influence-functions-for-penalized-M-estimators/10.3150/16-BEJ841.pdf?utm_source=chatgpt.com "Influence functions for penalized M-estimators - Project Euclid"
[9]: https://archive.ics.uci.edu/dataset/291/airfoil%2Bself%2Bnoise?utm_source=chatgpt.com "Airfoil Self-Noise - UCI Machine Learning Repository"
[10]: https://archive.ics.uci.edu/ml/datasets/yacht%2Bhydrodynamics?utm_source=chatgpt.com "Yacht Hydrodynamics - UCI Machine Learning Repository"
[11]: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html?utm_source=chatgpt.com "fetch_california_housing — scikit-learn 1.7.1 documentation"
[12]: https://saeb.feaa.uaic.ro/index.php/saeb/article/download/1656/312/3265?utm_source=chatgpt.com "[PDF] The Effectiveness of the Huber's Weight on Dispersion and Tuning ..."
[13]: https://link.springer.com/article/10.1057/s41260-022-00258-0?utm_source=chatgpt.com "Efficient bias robust regression for time series factor models"
[14]: https://bcmi.sjtu.edu.cn/~blu/papers/2017/2017-5.pdf?utm_source=chatgpt.com "[PDF] Robust structured sparse representation via half-quadratic ..."
[15]: https://www.isee-ai.cn/~zhwshi/Research/PreprintVersion/Half-quadratic%20based%20Iterative%20Minimization%20for%20Robust%20Sparse%20Representation.pdf?utm_source=chatgpt.com "[PDF] Half-quadratic based Iterative Minimization for Robust Sparse ..."
[16]: https://en.wikipedia.org/wiki/Median_absolute_deviation?utm_source=chatgpt.com "Median absolute deviation"
[17]: https://search.r-project.org/CRAN/refmans/rQCC/html/MAD.html?utm_source=chatgpt.com "Median absolute deviation (MAD) - R"
[18]: https://www.kaggle.com/datasets/fedesoriano/airfoil-selfnoise-dataset/data?utm_source=chatgpt.com "NASA Airfoil Self-Noise Dataset - Kaggle"
[19]: https://medium.com/%40shamdeepvk/a-regression-model-analysis-of-the-nasa-airfoil-self-noise-dataset-1d887ee7578b?utm_source=chatgpt.com "A regression model analysis of the NASA Airfoil Self-Noise Dataset."
[20]: https://www.kxy.ai/reference/latest/applications/illustrations/yacht.html?utm_source=chatgpt.com "Yacht Hydrodynamics (UCI, Regression, n=308, d=6)"
[21]: https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html?utm_source=chatgpt.com "The California housing dataset — Scikit-learn course"
[22]: https://codesignal.com/learn/courses/deep-dive-into-numpy-and-pandas-with-housing-data/lessons/exploring-the-california-housing-dataset-an-introduction-to-dataset-characteristics-and-basic-visualizations?utm_source=chatgpt.com "Exploring the California Housing Dataset - CodeSignal"
[23]: https://books.google.com/books/about/Information_Theoretic_Learning.html?id=oJSkBXWctsgC&utm_source=chatgpt.com "Information Theoretic Learning: Renyi's Entropy and Kernel ..."
[24]: https://scispace.com/pdf/information-theoretic-learning-5atkmbyehj.pdf?utm_source=chatgpt.com "[PDF] Information Theoretic Learning - SciSpace"
[25]: https://arxiv.org/pdf/2104.04416?utm_source=chatgpt.com "[PDF] Concentration study of M-estimators using the influence function."
