# CIE 457 — Course Project
## MLE Estimation and Estimator Distribution in Linear Regression

**Khaled Mahmoud (202200282) · Ahmed Gamal (202200133)**  
**Dr. Mahmoud Abdelaziz · Spring 2026**

---

## Setup

```
n         = 100
beta_true = [2.0, 1.5, -0.8]
sigma2    = 1.0
X shape   = (100, 3)   [intercept, x1, x2]
sigma_i2  = linspace(0.5, 3.0, 100)   [hetero variances]
N_sim     = 5000
```

---

## 3.1 MLE and Model Formulation

### Model

$$y = X\beta + \epsilon$$

### 3.1.1 Likelihood Functions

**Homoscedastic** — $\epsilon \sim \mathcal{N}(0, \sigma^2I)$:

$$\ell(\beta) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\|y - X\beta\|^2$$

**Heteroscedastic** — $\epsilon \sim \mathcal{N}(0, \Sigma)$:

$$\ell(\beta) = -\frac{n}{2}\log(2\pi) - \frac{1}{2}\log|\Sigma| - \frac{1}{2}(y-X\beta)^T\Sigma^{-1}(y-X\beta)$$

**Results:**
```
Homo  log-L at true beta : -150.300
Hetero log-L at true beta: -153.875
```

---

### 3.1.2 MLE Estimators

**OLS** (from homo MLE — set dℓ/dβ = 0):

$$\hat{\beta}_{OLS} = (X^TX)^{-1}X^Ty$$

**WLS** (from hetero MLE — set dℓ/dβ = 0):

$$\hat{\beta}_{WLS} = (X^T\Sigma^{-1}X)^{-1}X^T\Sigma^{-1}y$$

**Results:**
```
True beta : [ 2.0000   1.5000  -0.8000]
OLS       : [ 2.0928   1.6907  -0.9721]
WLS       : [ 2.1380   1.5742  -0.8287]

log-L at OLS: -147.396 ≥ -150.300 ✓  (OLS is the MLE)
```

---

### 3.1.3 Model Assumptions, Randomness, Distribution

**Assumptions:**
```
1. y = X @ beta + epsilon       (linearity)
2. X is fixed and known         (non-random)
3. epsilon ~ N(0, sigma2*I)     (homo)
   OR epsilon ~ N(0, Sigma)     (hetero)
4. X^TX is invertible           (no multicollinearity)
```

**Why β̂ is a Random Variable:**

$$\hat{\beta} = \beta + \underbrace{(X^TX)^{-1}X^T\epsilon}_{\text{random part}}$$

X fixed, β fixed → β̂ changes with every ε

```
Run 1: beta_hat = [ 1.9252  1.4593 -0.7344]
Run 2: beta_hat = [ 2.0420  1.5623 -0.8165]
Run 3: beta_hat = [ 1.9482  1.4262 -0.8433]
Run 4: beta_hat = [ 2.0608  1.5262 -0.6703]
Run 5: beta_hat = [ 2.1097  1.5490 -0.7826]
```

**Theoretical Distribution:**

| Case | Distribution | Cov Trace |
|---|---|---|
| OLS (homo) | $\mathcal{N}(\beta, \sigma^2(X^TX)^{-1})$ | 0.0341 |
| WLS (hetero, correct) | $\mathcal{N}(\beta, (X^T\Sigma^{-1}X)^{-1})$ | 0.0473 |
| OLS (hetero, misspecified) | $\mathcal{N}(\beta, (X^TX)^{-1}X^T\Sigma X(X^TX)^{-1})$ | 0.0592 |

> **Note:** OLS homo trace < WLS hetero trace because they operate under different noise levels (σ²=1.0 vs mean=1.75). Fair comparison: WLS (0.0473) vs OLS sandwich (0.0592) — both under same hetero noise.

---

## 3.2 Linear Regression under Homoscedastic Noise

### Single Sample OLS

```
                    b0       b1       b2
True beta       2.0000   1.5000  -0.8000
OLS estimate    2.0451   1.5981  -0.9300
Error           0.0451   0.0981  -0.1300
Theoretical std [0.1010  0.1174  0.1007]

epsilon mean : 0.0293  (expect ≈ 0)
epsilon std  : 0.8665  (expect ≈ 1.0)
```

### Monte Carlo Results (N_sim = 5000)

```
                         b0         b1         b2
True beta            2.0000     1.5000    -0.8000
Empirical Mean       1.9988     1.5000    -0.8038   ← unbiased ✓
Theoretical Mean     2.0000     1.5000    -0.8000
Empirical Std        0.1026     0.1148     0.0999   ← matches theory ✓
Theoretical Std      0.1010     0.1174     0.1007

Empirical Covariance:
[[ 0.0105  0.0014 -0.0003]
 [ 0.0014  0.0132 -0.0003]
 [-0.0003 -0.0003  0.0100]]

Theoretical Covariance:
[[ 0.0102  0.0016 -0.0004]
 [ 0.0016  0.0138 -0.0004]
 [-0.0004 -0.0004  0.0101]]
```

**Key Findings:**
- Empirical Mean ≈ True β → OLS is **unbiased** ✓
- Empirical Std ≈ Theoretical Std → variance formula correct ✓
- KDE matches N(β, σ²(XᵀX)⁻¹) → **Gaussian shape confirmed** ✓

---

## 3.3 Linear Regression under Heteroscedastic Noise

$\sigma^2_i \in [0.5, 3.0]$ — both OLS and WLS compared over 5000 runs

### Monte Carlo Results

```
                              b0         b1         b2
True beta                 2.0000     1.5000    -0.8000

── OLS (misspecified) ──
Empirical Mean            1.9978     1.5010    -0.8031   ← unbiased ✓
Empirical Std             0.1340     0.1540     0.1344   ← larger
Theoretical Std           0.1335     0.1531     0.1338

── WLS (correct model) ──
Empirical Mean            1.9980     1.4996    -0.8038   ← unbiased ✓
Empirical Std             0.1178     0.1372     0.1199   ← smaller ✓
Theoretical Std           0.1188     0.1374     0.1195
```

### Efficiency Summary

```
Metric                          OLS          WLS
Std of b0 (empirical)        0.1340       0.1178
Std of b1 (empirical)        0.1540       0.1372
Std of b2 (empirical)        0.1344       0.1199
Cov Trace (theoretical)      0.0592       0.0473
Efficiency gain               ——          1.25× lower variance

Conclusion:
- Both OLS and WLS are unbiased ✓
- WLS has 1.25× smaller total variance
- OLS ignores Sigma → wastes information → inefficient
- WLS uses Sigma^{-1} as weights → optimal → efficient
```

---

## 3.4 Inference from Estimator Distribution

### 3.4.1 Confidence Intervals

**Formula:**
$$\hat{\beta}_j \pm 1.96 \times \sqrt{\sigma^2[(X^TX)^{-1}]_{jj}}$$

where z = 1.96 = `stats.norm.ppf(0.975)`

**Results:**
```
               Estimate      Lower      Upper  True β   Inside?
beta_0           2.0174     1.8195     2.2154  2.0000       ✓
beta_1           1.3935     1.1633     1.6237  1.5000       ✓
beta_2          -0.8554    -1.0528    -0.6581 -0.8000       ✓

Coverage Verification (over 1000 samples):
beta_0  94.6%  (expect 95%) ✓
beta_1  94.6%  (expect 95%) ✓
beta_2  94.2%  (expect 95%) ✓
```

---

### 3.4.2 Prediction Intervals

**Formula:**
$$\hat{y}_{new} \pm 1.96 \times \sqrt{\sigma^2(1 + x_{new}^T(X^TX)^{-1}x_{new})}$$

**For x_new = [1.0, 0.5, -0.3]:**
```
y_hat (predicted) : 2.9708
y_true (no noise) : 2.9900

                         Lower    Upper    Width
95% CI (mean)           2.7189   3.2227   0.5037
95% PI (new obs)        0.9947   4.9469   3.9522

Variance Breakdown:
beta uncertainty : 0.0165
new noise        : 1.0000
total (PI)       : 1.0165

PI is wider than CI by factor: 7.85×
```

**Why PI > CI always:**
```
CI → uncertainty in beta_hat only    = 0.0165
PI → uncertainty in beta_hat + noise = 1.0165
As n → ∞: CI → 0, but PI stays wide (irreducible noise)
```

---

## 3.5 Real Data Application — Longley Dataset

### Dataset
```
Shape   : (16, 7)
Years   : 1947–1962
y       : TOTEMP (total employment)
Features: GNP, UNEMP, POP
X shape : (16, 4)  [intercept + 3 features]
```

### OLS Results
```
Coefficient    Value
intercept    66,157.75   (baseline)
GNP           0.0476     (economy grows → more employment ✓)
UNEMP        -0.3854     (more unemployed → less employment ✓)
POP          -0.1539     (multicollinearity with GNP)

R² = 0.9812  (model explains 98.12% of variance ✓)
```

### Residual Analysis
```
Breusch-Pagan Test:
BP statistic : 0.0974
p-value      : 0.7550
=> p > 0.05: No strong heteroscedasticity → OLS ok
```

### OLS vs WLS Comparison
```
               OLS coef    OLS std    WLS coef    WLS std
intercept    66157.75    23855.70    63864.78    11472.36
GNP              0.0476      0.0170      0.0459      0.0082
UNEMP           -0.3854      0.3315     -0.4075      0.1585
POP             -0.1539      0.2664     -0.1282      0.1280

R² OLS : 0.9812
R² WLS : 0.9812

Efficiency (Cov trace):
OLS trace : 5.69e+08
WLS trace : 1.32e+08
WLS is 4.33× more efficient than OLS
```

---

## Bonus — Fisher Information, CRLB, Sufficient Statistics

### Part A — Fisher Information & CRLB

**Step 1 — Log-Likelihood:**
$$\ell(\beta) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}(y-X\beta)^T(y-X\beta)$$

**Step 2 — Score Function:**
$$\frac{\partial\ell}{\partial\beta} = \frac{1}{\sigma^2}X^T(y-X\beta)$$

**Step 3 — Hessian:**
$$\frac{\partial^2\ell}{\partial\beta\partial\beta^T} = -\frac{1}{\sigma^2}X^TX$$

**Step 4 — Fisher Information Matrix:**
$$\mathcal{I}(\beta) = -E\left[\frac{\partial^2\ell}{\partial\beta\partial\beta^T}\right] = \frac{1}{\sigma^2}X^TX$$

**Step 5 — CRLB:**
$$\text{Var}(\hat{\beta}) \geq \mathcal{I}(\beta)^{-1} = \sigma^2(X^TX)^{-1}$$

**Step 6 — OLS achieves CRLB:**
$$\text{Cov}(\hat{\beta}_{OLS}) = \sigma^2(X^TX)^{-1} = \text{CRLB} \implies \text{OLS is MVUE}$$

**Results:**
```
I(beta) = (1/sigma2) * X^T X:
[[100.00  -11.56    3.40]
 [-11.56   73.93    2.34]
 [  3.40    2.34   98.89]]

CRLB = OLS Covariance = sigma2*(X^TX)^{-1}:
[[ 0.0102  0.0016 -0.0004]
 [ 0.0016  0.0138 -0.0004]
 [-0.0004 -0.0004  0.0101]]

Cov(OLS) - CRLB:
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]  ← zero matrix ✓

Diagonal comparison:
           CRLB      OLS Cov   Match
beta_0   0.010199   0.010199     ✓
beta_1   0.013790   0.013790     ✓
beta_2   0.010135   0.010135     ✓

OLS achieves CRLB: True ✓
=> OLS is the MVUE
```

---

### Part B — Sufficient Statistics

**Factorization Theorem:**
$$p(y|\beta,\sigma^2) = h(y) \cdot g(T(y), \beta, \sigma^2)$$

**Expand the Quadratic:**
$$(y-X\beta)^T(y-X\beta) = \underbrace{y^Ty}_{T_2} - 2\beta^T\underbrace{X^Ty}_{T_1} + \beta^TX^TX\beta$$

**Likelihood after Expansion:**
$$p(y|\beta,\sigma^2) = \underbrace{1}_{h(y)} \cdot \underbrace{\exp\!\left(-\frac{1}{2\sigma^2}\left[T_2 - 2\beta^TT_1 + \beta^TX^TX\beta\right]\right)}_{g(T_1,T_2,\beta,\sigma^2)}$$

**Sufficient Statistics:**
$$\boxed{T_1 = X^Ty \quad \text{sufficient for } \beta}$$
$$\boxed{T_2 = y^Ty \quad \text{sufficient for } \sigma^2}$$

**OLS from T₁ only:**
$$\hat{\beta}_{OLS} = (X^TX)^{-1}X^Ty = (X^TX)^{-1}T_1$$

**Data Compression:**
$$100 \text{ observations} \xrightarrow{T_1 = X^Ty} 3 \text{ numbers — zero information loss}$$

**Results:**
```
T1 = X^T y = [186.4332   98.5303  -85.0521]
T2 = y^T y = 750.4397

Verification — OLS from T1 only:
beta from T1 : [ 2.092793  1.690720  -0.972142]
beta from y  : [ 2.092793  1.690720  -0.972142]
Match        : True ✓

Verification — sigma2 from T1 and T2 only:
sigma2 from T1,T2 : 1.110035
sigma2 from y     : 1.110035
Match             : True ✓
```

---

## Key Formulas Summary

| Formula | Equation |
|---|---|
| OLS | $\hat{\beta} = (X^TX)^{-1}X^Ty$ |
| WLS | $\hat{\beta} = (X^T\Sigma^{-1}X)^{-1}X^T\Sigma^{-1}y$ |
| Cov OLS (homo) | $\sigma^2(X^TX)^{-1}$ |
| Cov WLS (hetero) | $(X^T\Sigma^{-1}X)^{-1}$ |
| Cov OLS (sandwich) | $(X^TX)^{-1}X^T\Sigma X(X^TX)^{-1}$ |
| 95% CI | $\hat{\beta}_j \pm 1.96\sqrt{\sigma^2[(X^TX)^{-1}]_{jj}}$ |
| 95% PI | $\hat{y}_{new} \pm 1.96\sqrt{\sigma^2(1 + x_{new}^T(X^TX)^{-1}x_{new})}$ |
| Fisher Info | $\mathcal{I}(\beta) = \frac{1}{\sigma^2}X^TX$ |
| CRLB | $\mathcal{I}(\beta)^{-1} = \sigma^2(X^TX)^{-1}$ |
| T1 (sufficient) | $X^Ty$ |
| T2 (sufficient) | $y^Ty$ |

---

## Key Conclusions

```
1. OLS = MLE under homo noise
   log-L at OLS ≥ log-L at true beta ✓

2. β̂ is a random variable
   β̂ = β + (X^TX)^{-1}X^T ε → changes with every ε

3. Distribution confirmed by simulation
   Empirical ≈ Theoretical N(β, σ²(X^TX)^{-1}) ✓
   KDE matches Gaussian PDF ✓

4. Under hetero noise:
   OLS → unbiased but inefficient (sandwich cov)
   WLS → unbiased and efficient (1.25× better)

5. CI: captures uncertainty in β̂ (narrow)
   PI: captures uncertainty in β̂ + future noise (wide, 7.85×)
   Coverage: 94.2–94.6% ≈ 95% ✓

6. Real data (Longley):
   R² = 0.9812, BP p = 0.7550 → OLS adequate
   WLS still 4.33× more efficient

7. Bonus:
   OLS achieves CRLB → MVUE ✓
   T1 = X^Ty sufficient for β (100 obs → 3 numbers) ✓
```
