# Assignment 17.1 — Comparing Classifiers (Bank Marketing)

## Business objective

Build and compare classification models that rank prospective customers by their likelihood of subscribing to a Portuguese bank term deposit (`y = yes`). The business goal is to improve campaign conversion while reducing agent time spent on low-propensity leads.

A false positive wastes a call; a false negative misses a potential subscription. The operating threshold should therefore reflect call capacity, call cost, and expected subscription value.

## Dataset

- **Source:** [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **File:** `data/bank-additional-full.csv` (downloaded automatically from UCI by the notebook if it is not present locally)
- **Rows:** 41,188
- **Predictors:** 20 original features plus binary target `y`
- **Target balance:** approximately 88.7% `no` and 11.3% `yes`

The notebook downloads the original UCI archive on first run if the CSV is absent. The CRISP-DM paper is cited in the notebook and linked in References; the raw PDF and dataset archive are not committed to keep the repository lightweight.

## Critical modeling decision: exclude `duration`

The `duration` field records the length of the current call. It is not known before the call begins and is strongly associated with the outcome. The primary models exclude it to prevent target leakage and to preserve the intended pre-call use case. The notebook includes an explicitly labelled diagnostic comparison with `duration`; that result must not be presented as deployable performance.

## Methodology

1. **Business understanding:** define the call-cost / missed-opportunity trade-off.
2. **Data understanding:** inspect types, missing values, `unknown` categories, class balance, duplicates, and feature distributions.
3. **EDA:** demographic, campaign-history, contact, and macroeconomic plots; numeric correlation heatmap.
4. **Preprocessing:** stratified train/test split; median imputation and standard scaling for numeric variables; most-frequent imputation and one-hot encoding for categorical variables.
5. **Leakage control:** preprocessing is encapsulated in scikit-learn pipelines and is fitted within cross-validation folds.
6. **Models:** K-Nearest Neighbors, Logistic Regression, Decision Tree, and Support Vector Classifier.
7. **Evaluation:** majority-class dummy baseline, accuracy, precision, recall, F1, ROC-AUC, and average precision (PR-AUC).
8. **Tuning:** 5-fold stratified cross-validation with ROC-AUC as the primary selection metric. SVC tuning uses a stratified 6,000-row training subset to keep RBF-kernel search practical; the selected configuration is refit on the full training split.
9. **Interpretation:** logistic-regression coefficients / odds ratios and decision-tree feature importance / top-level rules.
10. **Recommendations:** propensity ranking, capacity-aware thresholds, contact caps, controlled pilots, and monitoring.

## Why accuracy is insufficient

A classifier that predicts `no` for every customer would achieve approximately 88.7% accuracy but identify no subscribers (zero positive-class recall). ROC-AUC assesses ranking across thresholds; PR-AUC is particularly useful for the minority positive class. Precision, recall, and F1 describe threshold-specific trade-offs.

## Repository structure

```text
assignment_17_1_bank_marketing/
├── data/
│   ├── bank-additional-full.csv
│   └── bank-additional-names.txt
├── notebooks/
│   └── prompt_III_completed.ipynb
├── reports/
├── CRISP-DM-BANK.pdf
├── README.md
└── requirements.txt
```

## Run the notebook

Python 3.10+ is recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
jupyter lab
```

Open `notebooks/prompt_III_completed.ipynb` and run all cells from the repository root.

## Findings to report

The notebook computes the results rather than hard-coding expected scores. Use its final holdout table to identify the strongest model, then discuss the precision/recall trade-off. The EDA examines patterns in customer segments, prior campaign outcomes, contact intensity, and macroeconomic indicators. These are historical associations and do not prove causal effects.

## Actionable recommendations

- Rank prospects by predicted subscription propensity and pilot a capacity-sized top-*k* calling list.
- Set the operating threshold using call cost, expected deposit value, and agent capacity.
- Prioritize prior successful responders where permitted by consent and contact policy.
- Evaluate diminishing returns from repeated contacts and implement a contact cap / suppression policy.
- Monitor conversion lift, cost per subscription, opt-outs, complaints, and model drift.
- Validate performance in a controlled A/B test before operational rollout.

## Limitations and responsible use

The data represents a historical campaign and may not generalize to current customers, products, or macroeconomic conditions. Campaign variables can encode prior selection decisions. Demographic differences should not be treated as causal or used as automatic exclusion criteria. Review privacy, consent, fairness, and applicable telemarketing rules before deployment.

## References

- [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- Moro, S., Laureano, R., & Cortez, P. (2011), *Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology*.
- [scikit-learn documentation](https://scikit-learn.org/stable/)