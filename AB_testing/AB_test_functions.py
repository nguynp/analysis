# A/B Testing Function for Numerical data
def AB_Test_Numerical(df_control, df_treatment, target, alpha=0.05):
    '''
    Normality test: Shaphiro Wilk Test
    Homogeneity test: Levene's Test
    Hypothesis test: t-test or Mann-Whiteney U Test
    *applying for continuous variables*

    Parameters:
    -----------
    df_control: pd.DataFrame
        Dataframe of control group
    df_treatment: pd.DataFrame
        Dataframe of treatment group - apply new change(s)
    target: str
        Column name of df - target variable of the test
    alpha: float (0.05 by default)
        Level of significance (common <= alpha)

    Returns:
    -----------
    Testing results including reject or not reject null hypothesis
    '''
    # Packages
    from scipy.stats import shapiro
    import scipy.stats as stats

    ### Split A/B
    groupA = df_control[target]
    groupB = df_treatment[target]
    
    ### Testing the Normality Assumption
    print("### Testing the Normality Assumption for "+target+" ###\n")
    p_shapiro_A = shapiro(groupA)[1]
    p_shapiro_B = shapiro(groupB)[1]
    # H0: Distribution is Normal
    # H1: Distribution is not Normal
    
    if (p_shapiro_A>alpha) & (p_shapiro_B>alpha):
        print(f"Shaphiro Wilk Test resulted in p > {alpha} for both GroupA and GroupB,\n which indicates that H0 cannot be rejected. Thus GroupA and GroupB are likely to normal distribution.\n")
        
        # Parametric Test
        print("\n###Testing the Homogeneity Assumption for "+target+" ###\n")
        # Assumption: Homogeneity of variances
        p_levene = stats.levene(groupA, groupB)[1]
        # H0: Homogeneity
        # H1: Heterogeneous
        
        if p_levene > alpha:
            # Homogeneity
            print(f"\nLevene's Test for Homogeneity resulted in p = {p_levene} > {alpha}, which indicates that H0 cannot be rejected.\n")
            print(" Thus, variances of GroupA and GroupB are equal.","\n")

            print("\n### Testing Hypothesis for "+target+" with t-test & equal variances###")
            p_ttest = stats.p_ttest_ind(groupA, groupB, equal_var=True)[1]
            # H0: M1 == M2 
            # H1: M1 != M2
        else:
            # Heterogeneous
            print(f"Levene's Test for Homogeneity resulted in p = {p_levene} < {alpha}, which indicates that H0 is rejected.")
            print("Thus, variances of GroupA and GroupB are equal.")

            print("\n### Testing Hypothesis for "+target+" with t-test & unequal variances###\n")
            p_ttest = stats.p_ttest_ind(groupA, groupB, equal_var=False)[1]
            # H0: M1 == M2
            # H1: M1 != M2
    else:
        print(f"Shaphiro Wilk Test resulted in p < {alpha} for one or both GroupA and GroupB, which indicates that H0 is rejected.\nThus GroupA and GroupB are not likely to normal distribution.\n")
        # Non-Parametric Test
        print("\n### Testing Hypothesis for "+target+" with Mann-Whiteney U Test ###\n")
        p_ttest = stats.mannwhitneyu(groupA, groupB)[1] 
        # H0: M1 == M2 
        # H1: M1 != M2 
    
    if p_ttest < alpha:
        print(f'Hypothesis test result in p = {p_ttest:.3f} < {alpha}, which indicates that H0 is rejected.\nThus, mean of {target} between GroupA & Group B are not similar.')
    else:
        print(f'Hypothesis test result in p = {p_ttest:.3f} > {alpha}, which indicates that H0 cannot be rejected.\nThus, mean of {target} between GroupA & Group B are not likely unsimilar.')

# A/B Testing Function for Categorical data (binary)
def AB_Test_Categorical(df_control, df_treatment, target, condition, ztest_proportions = True, chisquare_test = True, alpha=0.05):
    '''
    Hypothesis test: two-sample Z Test for proportions and/or Chi-square Test
    *applying for binary categorical variables (2-dimension), thus will be treated as success or not*

    Parameters:
    -----------
    df_control: pd.DataFrame
        Dataframe of control group
    df_treatment: pd.DataFrame
        Dataframe of treatment group - apply new change(s)
    target: str
        Column name of df - target variable of the test
    condition: str/ bool
        condition of the target to consider as a success
    ztest_proportions: bool
        whether to performe this test
    chisquare_test: bool
        whether to performe this test
    alpha: float (0.05 by default)
        Level of significance (common <= alpha)

    Returns:
    -----------
    Testing results including reject or not reject null hypothesis
    '''
    import numpy as np
    
    # calculate all observations & frequencies
    nobs_groupA, nobs_groupB = len(df_control[target]), len(df_treatment[target])
    successes_groupA = len(df_control[df_control[target]==condition])
    successes_groupB = len(df_treatment[df_treatment[target]==condition])

    if ztest_proportions:

        from statsmodels.stats.proportion import proportions_ztest

        print("\n### Testing Hypothesis for "+target+" with two-sample Z Test for proportions ###\n")
        count = np.array([successes_groupA, successes_groupB])
        nobs = np.array([nobs_groupA, nobs_groupB])
        p_prop_ztest = proportions_ztest(count=count, nobs=nobs)[1]
        if p_prop_ztest > alpha:
            print(f"Hypothesis test result in p = {p_prop_ztest:.3f} > {alpha}, which indicates that H0 cannot be rejected.\nThus, the two proportions might be the same.\n")
        else:
            print(f"Hypothesis test result in p = {p_prop_ztest:.3f} < {alpha}, which indicates that H0 is rejected.\n Thus, the two proportions might be different.\n")
        
    if chisquare_test:

        import scipy.stats as stats

        print("\n### Testing Hypothesis for "+target+" with Chi-square Test ###\n")
        rxc_table = np.array([[successes_groupA, nobs_groupA - successes_groupA],
                              [successes_groupB, nobs_groupB - successes_groupB]])
        p_chi2 = stats.chi2_contingency(observed=rxc_table)[1]
        if p_chi2 > alpha:
            print(f"Hypothesis test result in p = {p_chi2:.3f} > {alpha}, which indicates that H0 cannot be rejected.\nThus, the two proportions might be the same.\n")
        else:
            print(f"Hypothesis test result in p = {p_chi2:.3f} < {alpha}, which indicates that H0 is rejected.\n Thus, the two proportions might be different.\n")
