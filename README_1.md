# CSCI 3329 — Homework 3 Report
## Comparing Classification Algorithms with Feature Selection

---

## 1. Dataset

| Property | Value |
|---|---|
| **Name** | Credit Approval |
| **Source** | [UCI Machine Learning Repository #27](https://archive.ics.uci.edu/dataset/27/credit+approval) |
| **Samples** | 690 (raw) → ~653 after dropping missing rows |
| **Features** | 15 (mix of continuous and categorical) |
| **Classes** | 2 — `+` (approved) and `-` (denied) |
| **Citation** | Quinlan, J. (1987). *Credit Approval* [Dataset]. UCI ML Repository. https://doi.org/10.24432/C5FS30 |

The dataset contains anonymised credit card application records. All attribute names and values
have been replaced with meaningless symbols to protect confidentiality. It is a standard
binary-classification benchmark with a useful mix of continuous and nominal features.

**Class Distribution**

| Class | Count | % |
|---|---|---|
| `+` (approved) | ~307 | ~47% |
| `-` (denied) | ~346 | ~53% |

The classes are roughly balanced, so plain accuracy is an appropriate metric.

![Class Distribution](class_distribution.png)

---

## 2. Preprocessing

**Missing values** — The raw file uses `?` to denote missing entries (present in columns A1,
A2, A4, A5, A6, A7, and A14). All rows containing at least one missing value were dropped,
leaving ~653 complete records.

**Irrelevant columns** — None: all 15 columns are anonymised features with no ID or index column.

**Encoding** — Categorical columns (A1, A4, A5, A6, A7, A9, A10, A12, A13) were encoded
with `sklearn.preprocessing.LabelEncoder`. The target column A16 (`+`/`-`) was also label-encoded.

**Scaling** — All 15 features were standardised with `StandardScaler` (zero mean, unit
variance). This step is critical for KNN and MLP, which are sensitive to feature scale, and
has no effect on Gaussian NB.

**Fixed seed** — `random_state=17342` was used in every stochastic component.

---

## 3. Part 2 — Algorithm Comparison

**Evaluation protocol:** 10-fold cross-validation repeated 100 times (`RepeatedKFold` with
`n_splits=10`, `n_repeats=100`) → 1,000 accuracy values per algorithm.

| Algorithm | Mean Accuracy | Std |
|---|---|---|
| Linear Classifier | 0.8090 | 0.0621 |
| Logistic Regression | 0.8675 | 0.0401 |
| KNN | 0.8636 | 0.0396 |
| Gaussian NB | 0.8028 | 0.0451 |
| Neural Network | 0.8612 | 0.0402 |

![Part 2 Comparison](part2_comparison.png)

### Discussion

*(Replace the placeholder text below with your actual observations once you have numbers.)*

**Best-performing algorithm:** Logistic Regression or the Neural Network typically lead on this
dataset because the credit-approval decision boundary is approximately linear in the encoded
feature space, which favours linear models. If the Neural Network wins, it likely benefits from
capturing small non-linear interactions.

**Are the gaps meaningful?** Compare each pair of means relative to their standard deviations.
If |mean_A − mean_B| < max(std_A, std_B), the difference may not be statistically reliable.
A difference larger than the combined spread is more compelling.

**Dataset characteristics and their effect:**
- 653 samples is comfortable for all five algorithms; NB and Perceptron converge quickly
  while MLP benefits from the larger sample size relative to its parameter count.
- 15 features is manageable; KNN works well in low-to-moderate dimensions.
- The mix of continuous and categorical inputs after label encoding may slightly violate
  Gaussian NB's assumption that every feature is continuous and independent, which could
  explain any gap between NB and the other models.

---

## 4. Part 3 — Feature Selection

**Search method:** Forward Sequential Feature Selection (`sklearn.feature_selection.SequentialFeatureSelector`,
`direction='forward'`, `n_features_to_select='auto'`).

**Justification:** With m = 15 features, exhaustive search requires evaluating 2¹⁵ − 1 = 32,767
subsets × 1,000 CV rounds ≈ 33 million evaluations, which is impractical on a single machine.
Forward selection reduces this to O(m²) ≈ 120 model fits per algorithm while still finding a
good (if not always globally optimal) subset. The greedy strategy is well-suited here because
the feature set is small enough that it rarely misses important synergies.

**Note on runtime:** The inner CV loop for SFS uses `n_repeats=10` (100 CV rounds per
candidate subset) rather than 100, to keep total wall-clock time reasonable. The final accuracy
of the chosen subset is then re-evaluated with the full `n_repeats=100` protocol to ensure a
fair comparison with Part 2.

| Algorithm | Best Feature Subset | # Features | Mean Accuracy | Std |
|---|---|---|---|---|
| Linear Classifier | A1, A3, A5, A6, A7, A8, A11 | 7 | 0.8124 | 0.0764 |
| Logistic Regression | A1, A2, A3, A4, A5, A6, A7 | 7 | 0.8664 | 0.0412 |
| KNN | A1, A2, A3, A4, A7, A11, A13 | 7 | 0.8809 | 0.0387 |
| Gaussian NB | A4, A6, A7, A9, A10, A11, A15 | 7 | 0.8711 | 0.0402 |
| Neural Network | A1, A4, A5, A6, A7, A11, A12 | 7 | 0.8733 | 0.0401 |

![Part 3 Comparison](part3_comparison.png)

---

## 5. Discussion

### Part 2 vs Part 3

**Overall:** Feature selection improved accuracy for every algorithm except Logistic Regression, which dropped marginally by 0.0010 — effectively no change. The biggest winner was Gaussian NB (+0.0682), followed by KNN (+0.0174) and Neural Network (+0.0121).

**Gaussian NB** benefited the most from feature selection. Dropping 8 of 15 features (keeping A4, A6, A7, A9, A10, A11, A15) likely removed features that violated the independence assumption or introduced noise, bringing the model closer to its ideal conditions.

**KNN** also improved noticeably (+0.0174). This is expected — KNN is sensitive to irrelevant features because extra dimensions inflate distances uniformly, diluting the signal from informative ones. Reducing from 15 to 7 features sharpened the distance metric.

**Neural Network** improved slightly (+0.0121). The MLP can partially learn to ignore irrelevant features through its weights, but removing them explicitly still helped a little.

**Logistic Regression** was essentially unchanged (-0.0010), which makes sense — L2 regularisation already suppresses irrelevant features internally, so explicit feature selection adds little.

**Linear Classifier (Perceptron)** improved slightly (+0.0034) but remained the lowest performer and had the highest variance (std=0.0764), reflecting its lack of regularisation and sensitivity to data ordering.

**Best overall algorithm:** Logistic Regression on all features (0.8675) and KNN with selected features (0.8809) are the top performers. KNN with 7 features achieves the highest mean accuracy of any model in either part.

### Limitations and Ideas for Improvement

- The greedy nature of forward selection can miss globally optimal subsets. A genetic
  algorithm or simulated annealing search could explore the space more thoroughly.
- Reducing `n_repeats` inside SFS speeds things up but introduces more variance in the
  feature-selection decision; increasing it would make the selected subset more stable.
- Hyperparameter tuning (e.g., `k` for KNN, layer sizes for MLP) was not performed here;
  combining feature selection with grid search could improve results further.
- The dataset attributes are all anonymised, so domain knowledge cannot be used to
  guide feature selection or interpret which features drive predictions.

---

## 6. Reproduction

**Python version:** 3.10+

**Key libraries:**

```
scikit-learn>=1.3
pandas
numpy
matplotlib
ucimlrepo
```

**Install dependencies:**

```bash
pip install scikit-learn pandas numpy matplotlib ucimlrepo
```

**Run the notebook:**

```bash
jupyter notebook hw3_credit_approval.ipynb
```

Or run as a script (if you export the notebook):

```bash
jupyter nbconvert --to script hw3_credit_approval.ipynb
python hw3_credit_approval.py
```

All random states are fixed to `17342` throughout the notebook for full reproducibility.
