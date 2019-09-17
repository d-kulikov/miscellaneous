# model-assessment

## statistical_comparison
Statistical comparison of performances of a model and some benchmark (e.g., a baseline model or naive approach) or two
competing models. Applied to binary classification problems. Only point estimates (calculated metrics such as AUC) are not enough.
It is important to make sure that one model is statistically better, not just due to a fluctuation. If it is not the case, it
does not make sense to use a more sophisticated model. Binomial tests (e.g., the chi-squared test) are also arbitrary at 
some extent because they require a hard threshold for determining a positive or negative outcome. This function works directly 
with probability estimates.
At first logarithmic errors are calculated (based on the formula of binary cross-entropy). Then two statistical tests 
for two paired samples are applied (T-test and Wilcoxon test) to these vectors of errors. High p-values (more than 0.05) 
indicate that the better model (with a better point estimate) is not better in statistical terms and may not be necessary
prefered.
