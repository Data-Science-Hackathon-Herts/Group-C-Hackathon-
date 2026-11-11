# Mock Data Science Challenge – Dataset A

Predicting seasonal flu vaccine uptake, and choosing the predictions that give the **lowest cost for the town**.

**Team:** _<team name>_
**Members:** _<names>_
**Colab:** _<link with Editor rights>_

---

## Problem

We are given 29 survey features about each respondent: opinions, behaviours, health and demographics. We predict whether each respondent received the seasonal flu vaccine (`seasonal_vaccine`, 1 = yes).

The task is scored by a measure specific to the task, not by accuracy. Two costs are given:

| Item | Cost |
|---|---|
| Seasonal (H3N2) vaccine | $300 |
| Hospitalisation for H3N2 flu | $18,500 |

## Evaluation metric

The model decides who to contact and offer a vaccine. Someone predicted as already vaccinated (1) is not contacted.

| | Actually vaccinated (1) | Actually not vaccinated (0) |
|---|---|---|
| **Predict 1** | Correct – $0 | **FP: left unprotected, risk of hospitalisation – $18,500** |
| **Predict 0** | **FN: unnecessary vaccine – $300** | Correct – $0 |

**Total cost = 18,500 × FP + 300 × FN**

A false positive is about 62× more expensive than a false negative. So we predict 1 only when the model's probability is above
**18,500 / 18,800 ≈ 0.984**, not the usual 0.5. The baseline every model must beat is "offer everyone a vaccine" (predict all 0).

## Repository contents

```
├── Dataset_A_Complete.ipynb      # all the code: EDA, metric, 5 models, cost comparison, submission
├── dataset_A_training.csv        # labelled data (4,756 rows)
├── dataset_A_testing.csv         # test data, target hidden (4,749 rows = "the town")
├── results/                      # created by the notebook: one set of files per model
│   ├── <Model>_results.json
│   ├── <Model>_holdout_proba.csv
│   └── <Model>_test_predictions.csv
├── submissions/                  # created by the notebook: 5 files ranked by cost
└── README.md
```

## Notebook structure

| Section | Content |
|---|---|
| 1. Setup | Libraries and settings |
| 2. Load the data | Reads the two CSV files (uploads them in Colab if missing) |
| 3. Exploratory data analysis | Missing values, target balance, vaccination rate by feature, correlations, mutual information, train vs test check |
| 4. Evaluation metric | Cost matrix, cost-optimal threshold (0.984), baseline policies |
| 5. Pre-processing, split and CV | Option A (simple) vs option B (missing-aware), 80/20 hold-out split, 5-fold and repeated CV |
| 6. Logistic Regression | |
| 7. Random Forest | |
| 8. Neural Network (MLP) | |
| 9. Extra Trees | |
| 10. Support Vector Machine (RBF) | |
| 11. Final cost comparison | Estimated cost for the town, ranking, sensitivity analysis, ranked submission files |
| 12. Conclusions | |

Each model section (6–10) follows the same steps:

1. Compare pre-processing option A with option B.
2. Tune hyper-parameters (`RandomizedSearchCV`, log loss).
3. Plot cost against decision threshold.
4. Run 15-fold repeated cross-validation.
5. Validate on the hold-out set, with a confusion matrix.
6. Explain the model (coefficients or feature importance, plus permutation importance).
7. Save results and test predictions.

## Pre-processing

- **Option A – simple:** median / most-frequent imputation, one-hot encoding, `employment_sector` dropped (49% missing).
- **Option B – missing-aware:** missing-value indicator columns, age / education / income encoded as ordered ranks, and missing categories kept as their own `'Missing'` level.
  `employment_sector` is missing for people who are not employed, and a missing `health_insurance` answer is linked to lower vaccination.

Option B gave better log loss for all five models, so it was used for every final model.

## Results

The town is the 4,749 residents in the test set. The estimated town cost is the mean cost per person over 15 repeated-CV folds × 4,749.

| Rank | Model | Est. town cost | Saving vs vaccinating everyone |
|---|---|---|---|
| 1 | **Extra Trees** | ≈ $758,300 | ≈ $5,700 |
| 2 | Logistic Regression | ≈ $761,000 | ≈ $3,000 |
| 3 | SVM (RBF) | ≈ $761,000 | ≈ $3,000 |
| 4 | Neural Network (MLP) | ≈ $762,700 | ≈ $1,400 |
| 5 | Random Forest | ≈ $763,000 | ≈ $1,000 |
| – | Baseline: vaccinate everyone | ≈ $764,000 | – |

- All five models beat or match the baseline on every one of the 15 CV folds.
- With the default 0.5 threshold, every model would cost the town about **$10 million**, roughly 13–14× more.
- The main drivers of vaccination, consistent across all models: perceived risk of flu, doctor recommendation, age, and belief in vaccine effectiveness.
- Sensitivity analysis: if hospitalisation after a missed vaccine is not certain (e.g. a 5–10% chance), every model saves far more. In that case logistic regression or the neural network ranks first, so the ranking depends on how the cost is interpreted.

## How to run

**Google Colab**
1. Open `Dataset_A_Complete.ipynb` in Colab.
2. Choose **Runtime → Run all**.
3. When asked, upload `dataset_A_training.csv` and `dataset_A_testing.csv`.
4. The full run takes about 7–10 minutes. The ranked submissions are downloaded as `submissions.zip` at the end.

**Locally**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook Dataset_A_Complete.ipynb
```
Keep the two CSV files in the same folder as the notebook.

Before submitting, change `TEAM_NAME` in Section 11 and check that the file format matches the official submission instructions.

## Reproducibility

All splits, folds and models use `RANDOM_STATE = 42`, so re-running the notebook gives the same results.
