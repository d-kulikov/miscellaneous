def statistical_comparison( predictions_model, predictions_benchmark, actual_labels ) :
    
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
    
    Parameters
    predictions_model: predicted values of the model of main interest (better, more advanced model)
    predictions_benchmark: predicted values of the benchmark model
    actual_labels: actual values (should be coded as 1 and 0)
    
    Example
    statistical_comparison( Neural_network_probs, Logistic_regression_probs, Y )
    
    Note
    Since the tests are one-sided, they always test a better model against a worse one. The other way round does not make sense.
    """
    
    import numpy as np
    
    from scipy.stats import ttest_rel, wilcoxon
    
    # Calculates logarithmic errors of the main model
    Errors_model = np.where( actual_labels == 1, -np.log( predictions_model ), -np.log( 1 - predictions_model ) )
    
    # Calculates logarithmic errors of the benchmark model
    Errors_benchmark = np.where( actual_labels == 1, -np.log( predictions_benchmark ), -np.log( 1 - predictions_benchmark ) )
    
    # Applies a parametric test, takes 1/2 of the p-value because the test is one-sided
    print( 'T-test p-value =', ttest_rel( Errors_model, Errors_benchmark )[ 1 ] / 2 )
    
    # Applies a non-parametric test, takes 1/2 of the p-value because the test is one-sided
    print( 'Wilcoxon test p-value =' , wilcoxon( Errors_model, Errors_benchmark )[ 1 ] / 2 )