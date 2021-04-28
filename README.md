# Multiple Comparisons

The code in this repository is used to run numerical experiments evaluating the Benjamini-Hochberg and Benjamini-Yekutieli procedures in two educational data mining research scenarios.

# Requirements

A Python installation with NumPy, SciPy, statsmodels, and joblib is required.  The code has been tested on Python 3.7 and 3.8.

# Usage

Classification model comparisons, no false nulls, n = 5000:
```python
sim_data = multiple_comparisons.run_classifier_sims(
    classifier_sd=[0.5]*6, n_jobs=1, test_size=5000)
fdr_accuracy = multiple_comparisons.analyze_classifier_results(
    sim_data, classifier_sd=[0.5]*6, AUROC=false)
fdr_auroc = multiple_comparisons.analyze_classifier_results(
    sim_data, classifier_sd=[0.5]*6, AUROC=True)
```
Classification model comparisons, false nulls, n = 5000:
```python
sim_data = multiple_comparisons.run_classifier_sims(
    classifier_sd=[0.1, 0.1, 0.1, 0.5, 1, 2], n_jobs=1, test_size=5000)
fdr_accuracy = multiple_comparisons.analyze_classifier_results(
    sim_data, classifier_sd=[0.1, 0.1, 0.1, 0.5, 1, 2], AUROC=false)
fdr_auroc = multiple_comparisons.analyze_classifier_results(
    sim_data, classifier_sd=[0.1, 0.1, 0.1, 0.5, 1, 2], AUROC=True)
```
Sequence transitions, no false nulls, n = 200:
```python
sim_data = multiple_comparisons.run_classifier_sims(
    rate=0., n_jobs=1, num_trials=200)
fdr_marginal = multiple_comparisons.analyze_sequence_results(
    sim_data, dependence=False, L_star=False)
fdr_L_star = multiple_comparisons.analyze_sequence_results(
    sim_data, dependence=False, L_star=True)
```
Sequence transitions, false nulls, n = 200:
```python
sim_data = multiple_comparisons.run_classifier_sims(
    rate=0.05, n_jobs=1, num_trials=200)
fdr_marginal = multiple_comparisons.analyze_sequence_results(
    sim_data, dependence=False, L_star=False)
fdr_L_star = multiple_comparisons.analyze_sequence_results(
    sim_data, dependence=False, L_star=True)
```
