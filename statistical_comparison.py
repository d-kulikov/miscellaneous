def comparison_classification( predictions_model, predictions_benchmark, actual_labels ) :
    
    import numpy as np
    from scipy.stats import ttest_rel, wilcoxon
    
    """ Statistical comparison of performances of a model and some benchmark (e.g., a baseline model or naive approach) or two
    competing models. Applied to binary classification problems. Only point estimates (calculated metrics such as AUC) are not enough.
    It is important to make sure that one model is statistically better, not just due to a fluctuation. If it is not the case, it
    does not make sense to use a more sophisticated model. Binomial tests (e.g., the chi-squared test) are also arbitrary at 
    some extent because they require a hard threshold for determining a positive or negative outcome. This function works directly 
    with probability estimates.
    At first logarithmic errors are calculated (based on the formula of binary cross-entropy). Then two statistical tests 
    for two paired samples are applied (T-test and Wilcoxon test) to these vectors of errors. High p-values (more than 0.05) 
    indicate that the better model (with a better point estimate) is not better in statistical terms and may not be necessary
    prefered.
    
    predictions_model: predicted probabilities of the model of main interest (better, more advanced model)
    predictions_benchmark: predicted probabilities of the benchmark model
    actual_labels: actual values (must be coded as 1 and 0)
    
    Example:
    mymodel_pred = np.array([ 0.53, 0.18, 0.62, 0.44, 0.73, 0.21, 0.59, 0.34, 0.67, 0.14 ])
    luck = np.array([ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 ])
    label = np.array([ 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 ])
    comparison_classification( mymodel_pred, luck, label )
    
    Note:
    Since the tests are one-sided, they always test a better model against a worse one. """
    
    # Calculates logarithmic errors of the main model
    errors_model = np.where( actual_labels == 1, -np.log( predictions_model ), -np.log( 1 - predictions_model ) )
    
    # Calculates logarithmic errors of the benchmark model
    errors_benchmark = np.where( actual_labels == 1, -np.log( predictions_benchmark ), -np.log( 1 - predictions_benchmark ) )
    
    # Applies a parametric test, takes 1/2 of the p-value because the test is one-sided
    tpvalue = ttest_rel( errors_model, errors_benchmark )[ 1 ] / 2
    print( 'T-test p-value =', round( tpvalue, 4 ) )
    
    # Applies a non-parametric test, takes 1/2 of the p-value because the test is one-sided
    wpvalue = wilcoxon( errors_model, errors_benchmark )[ 1 ] / 2
    print( 'Wilcoxon test p-value =' , round( wpvalue, 4 ) )
    
    
    
    
def comparison_regression( predictions_model, predictions_benchmark, actual_labels ) :
    
    import numpy as np
    from scipy.stats import ttest_rel, wilcoxon
    
    """ Statistical comparison of performances of a model and some benchmark (e.g., a baseline model or naive approach) or two
    competing models. Applied to regression problems. Only point estimates (calculated metrics such as RMSE) are not enough.
    It is important to make sure that one model is statistically better, not just due to a fluctuation. If it is not the case, it
    does not make sense to use a more sophisticated model. 
    At first errors of both models are calculated. Then two statistical tests for two paired samples are applied 
    (T-test and Wilcoxon test) to these vectors of errors. High p-values (more than 0.05) indicate that 
    the better model (with a better point estimate) is not better in statistical terms and may not be necessary prefered.
    
    predictions_model: predicted values of the model of main interest (better, more advanced model)
    predictions_benchmark: predicted values of the benchmark model
    actual_labels: actual values
    
    Example:
    mymodel_pred = np.array([ -5.1, -4.2, -3.3, -2.4, -1.5, 1.6, 2.7, 3.8, 4.9, 5.0 ])
    luck = np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ])
    label = np.array([ -10, -8, -6, -4, -2, 2, 4, 6, 8, 10 ])
    comparison_regression( mymodel_pred, luck, label )
    
    Note:
    Since the tests are one-sided, they always test a better model against a worse one. """
    
    # Calculates errors of the main model
    errors_model = np.abs( mymodel_pred - actual_labels )
    
    # Calculates errors of the benchmark model
    errors_benchmark = np.abs( mymodel_pred - luck )
    
    # Applies a parametric test, takes 1/2 of the p-value because the test is one-sided
    tpvalue = ttest_rel( errors_model, errors_benchmark )[ 1 ] / 2
    print( 'T-test p-value =', round( tpvalue, 4 ) )
    
    # Applies a non-parametric test, takes 1/2 of the p-value because the test is one-sided
    wpvalue = wilcoxon( errors_model, errors_benchmark )[ 1 ] / 2
    print( 'Wilcoxon test p-value =' , round( wpvalue, 4 ) )
