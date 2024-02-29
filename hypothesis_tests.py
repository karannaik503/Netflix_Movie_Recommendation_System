import numpy as np
import scipy.stats as stats
from scipy.stats import norm
import math
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.descriptivestats import sign_test
from termcolor import colored

import pandas as pd
from scipy.stats import chisquare
from scipy.stats import chi2

from scipy.stats import chi2_contingency
import random
from runs_table import table

def left_tailed_z_test(sample_dataset, null_hypothesis_mean, pop_variance, alpha):
    z_critical = norm.ppf(alpha)
    SEM = pop_variance/math.pow(len(sample_dataset),0.5)
    z_value = (np.mean(sample_dataset)-null_hypothesis_mean)/SEM
    print("alpha: ", alpha)
    print("z-statistic, z-critical: %.2f, %.2f"%(z_value, z_critical))
    if z_value <= z_critical:
        print("%.2f < %.2f"%(z_value, z_critical))
        print(colored("left tailed test at alpha %.2f: \tHypothesis rejected | H0: mean >= %.1f"%(alpha,null_hypothesis_mean), 'red'))
    else:
        print("%.2f > %.2f"%(z_value, z_critical))
        print(colored("left tailed test at alpha %.2f:\t Hypothesis accepted | H0: mean >= %.1f"%(alpha,null_hypothesis_mean), 'green'))

def right_tailed_z_test(sample_dataset, null_hypothesis_mean, pop_variance, alpha):
    z_critical = norm.ppf(1-alpha)
    SEM = pop_variance/math.pow(len(sample_dataset),0.5)
    z_value = (np.mean(sample_dataset)-null_hypothesis_mean)/SEM
    print("alpha: ", alpha)
    print("z-statistic, z-critical: %.2f, %.2f"%(z_value, z_critical))
    if z_value >= z_critical:
        print("%.2f > %.2f"%(z_value, z_critical))
        print(colored("right tailed test at alpha %.2f:\t Hypothesis rejected | H0: mean <= %.1f"%(alpha,null_hypothesis_mean), 'red'))
    else:
        print("%.2f < %.2f"%(z_value, z_critical))
        print(colored("right tailed test at alpha %.2f:\t Hypothesis accepted | H0: mean <= %.1f"%(alpha,null_hypothesis_mean), 'green'))

def two_tailed_z_test(sample_dataset, null_hypothesis_mean, pop_variance, alpha):
    z_critical = norm.ppf(1 - alpha/2)
    SEM = pop_variance/math.pow(len(sample_dataset),0.5)
    z_value = (np.mean(sample_dataset)-null_hypothesis_mean)/SEM
    print("alpha: ", alpha)
    print("z-statistic, z-critical: %.2f, %.2f"%(z_value, z_critical))
    if abs(z_value) >= z_critical:
        print("| %.2f | > %.2f"%(z_value, z_critical))
        print(colored("two tailed test at alpha %.2f:\t Hypothesis rejected | H0: mean = %.1f"%(alpha,null_hypothesis_mean), 'red'))
    else:
        print("| %.2f | < %.2f"%(z_value, z_critical))
        print(colored("two tailed test at alpha %.2f:\t Hypothesis accepted | H0: mean = %.1f"%(alpha,null_hypothesis_mean), 'green'))


def two_sample_test(sample_dataset, sample_dataset_2, alpha):
    SE1 = np.std(sample_dataset, ddof=1)
    SE2 = np.std(sample_dataset_2, ddof=1)
    n1 = len(sample_dataset)
    n2 = len(sample_dataset_2)
    SE = math.pow(SE1**2/n1 + SE2**2/n2, 0.5)
    z_critical = norm.ppf(1 - alpha/2)

    z_value = (np.mean(sample_dataset)-np.mean(sample_dataset_2))/SE
    print("alpha: ", alpha)
    print("z-statistic, z-critical: %.2f, %.2f"%(z_value, z_critical))
    if abs(z_value) >= z_critical:
        print("| %.2f | > %.2f"%(z_value, z_critical))
        print(colored("two sample z test at alpha %.2f:\t Hypothesis rejected | H0: mean1 = mean2"%alpha, 'red'))
    else:
        print("| %.2f | < %.2f"%(z_value, z_critical))
        print(colored("two sample z test at alpha %.2f:\t Hypothesis accepted | H0: mean1 = mean2"%alpha, 'green'))

def left_tailed_t_test(sample_dataset, null_hypothesis_mean, alpha):
    df = len(sample_dataset) - 1
    t_critical = stats.t.ppf(alpha, df)
    SEM = np.std(sample_dataset)/math.pow(df+1, 0.5)
    t_value = (np.mean(sample_dataset)-null_hypothesis_mean)/SEM
    print("Sample size: ", df+1)
    print("Degrees of freedom: ", df)
    print("alpha: ", alpha)
    print("t-statistic, t-critical: %.2f, %.2f"%(t_value, t_critical))
    if t_value <= t_critical:
        print("%.2f < %.2f"%(t_value, t_critical))
        print(colored("left tailed t-test at alpha %.2f:\t Hypothesis rejected | H0: mean >= %.1f"%(alpha,null_hypothesis_mean), 'red'))
    else:
        print("%.2f > %.2f"%(t_value, t_critical))
        print(colored("left tailed t-test at alpha %.2f:\t Hypothesis accepted | H0: mean >= %.1f"%(alpha,null_hypothesis_mean), 'green'))

def right_tailed_t_test(sample_dataset, null_hypothesis_mean, alpha):
    df = len(sample_dataset) - 1
    t_critical = stats.t.ppf(1 - alpha, df)
    SEM = np.std(sample_dataset)/math.pow(df+1, 0.5)
    t_value = (np.mean(sample_dataset)-null_hypothesis_mean)/SEM
    print("Sample size: ", df+1)
    print("Degrees of freedom: ", df)
    print("alpha: ", alpha)
    print("t-statistic, t-critical: %.2f, %.2f"%(t_value, t_critical))
    if t_value >= t_critical:
        print("%.2f > %.2f"%(t_value, t_critical))
        print(colored("right tailed t-test at alpha %.2f:\t Hypothesis rejected | H0: mean <= %.1f"%(alpha,null_hypothesis_mean), 'red'))
    else:
        print("%.2f < %.2f"%(t_value, t_critical))
        print(colored("right tailed t-test at alpha %.2f:\t Hypothesis accepted | H0: mean <= %.1f"%(alpha,null_hypothesis_mean), 'green'))

def two_tailed_t_test(sample_dataset, null_hypothesis_mean, alpha):
    df = len(sample_dataset) - 1
    t_critical = stats.t.ppf(1 - alpha/2, df)
    SEM = np.std(sample_dataset)/math.pow(df+1, 0.5)
    t_value = (np.mean(sample_dataset)-null_hypothesis_mean)/SEM
    print("Sample size: ", df+1)
    print("Degrees of freedom: ", df)
    print("alpha: ", alpha)
    print("t-statistic, t-critical: %.2f, %.2f"%(t_value, t_critical))
    if abs(t_value) >= t_critical:
        print("| %.2f | > %.2f"%(t_value, t_critical))
        print(colored("two tailed t-test at alpha %.2f:\t Hypothesis rejected | H0: mean = %.1f"%(alpha,null_hypothesis_mean), 'red'))
    else:
        print("| %.2f | < %.2f"%(t_value, t_critical))
        print(colored("two tailed t-test at alpha %.2f:\t Hypothesis accepted | H0: mean != %.1f"%(alpha,null_hypothesis_mean), 'green'))

def two_sample_t_value_test(sample_dataset, sample_dataset_2, alpha):
    print(sample_dataset)
    print(sample_dataset_2)
    n1 = len(sample_dataset)
    n2 = len(sample_dataset_2)
    df = n1+n2-2
    t_critical = stats.t.ppf(1 - alpha/2, df)

    SE1 = np.std(sample_dataset)
    SE2 = np.std(sample_dataset_2)
    
    sp = math.pow( ( (n1-1)*(SE1**2) + (n2-1)*(SE2**2) )/df, 0.5 )
    SE = sp * math.pow((1/n1 + 1/n2), 0.5)

    t_value = (np.mean(sample_dataset)-np.mean(sample_dataset_2))/SE
    print("Sample sizes: ", n1, n2)
    print("Degrees of freedom: ", df)
    print("alpha: ", alpha)
    print("t-statistic, t-critical: %.2f, %.2f"%(t_value, t_critical))
    if abs(t_value) >= t_critical:
        print("| %.2f | > %.2f"%(t_value, t_critical))
        print(colored("two sample t test at alpha %.2f:\t Hypothesis rejected | H0: mean1 = mean2"%alpha, 'red'))
    else:
        print("| %.2f | < %.2f"%(t_value, t_critical))
        print(colored("two sample t test at alpha %.2f:\t Hypothesis accepted | H0: mean1 = mean2"%alpha, 'green'))

def create_paired_sample(df, movie1, movie2):
    filtered_df = df[df['movieid'].isin([movie1, movie2])]
    pivot_df = filtered_df.pivot(index='userid', columns='movieid', values='rating').reset_index()
    pivot_df.columns = ['userid', 'movie1', 'movie2']
    result_df = pivot_df.dropna()
    # sample_df = result_df.sample(n=length, random_state=42)
    return result_df

def paired_sample_t_test(dataframe, alpha):
    df = dataframe.copy(deep = True)
    df['diff'] = df['movie1'] - df['movie2']
    df['diff2'] = (df['movie1'] - df['movie2'])**2
    d_sum = df['diff'].sum()
    d2_sum = df['diff2'].sum()
    n = len(df)
    d_mean = d_sum/n
    sd = math.pow((n*d2_sum - d_sum**2)/(n*(n-1)), 0.5)
    SEM = sd/math.pow(n,0.5)

    t_value = d_mean/SEM
    t_critical = stats.t.ppf(1 - alpha/2, n-1)
    print(df.to_string(index=False))
    df = df.drop('diff', axis=1)
    df = df.drop('diff2', axis=1)

    
    print("Sample size: ", n)
    print("Degrees of freedom: ", n-1)
    print("alpha: ", alpha)
    print("t-statistic, t-critical: %.2f, %.2f"%(t_value, t_critical))
    if abs(t_value) >= t_critical:
        print("| %.2f | > %.2f"%(t_value, t_critical))
        print(colored("paired sample t-test at alpha %.2f:\t Hypothesis rejected | H0: mean difference = 0"%alpha, 'red'))
    else:
        print("| %.2f | < %.2f"%(t_value, t_critical))
        print(colored("paired sample t-test at alpha %.2f:\t Hypothesis accepted | H0: mean difference != 0"%alpha, 'green'))

def proportion_test_one_sample(sample_data, rating, null_hypothesis_prop, alpha):
    check1 = len(sample_data)*null_hypothesis_prop
    check2 = len(sample_data)*(1- null_hypothesis_prop)
    print("Sample size: ", len(sample_data))
    print("np = %.2f, nq = %.2f\n"%(check1,check2))
    if(check1 < 5 or check2 < 5):
        print("Check failed. Can't perform proportion test")
        return
    z_critical = norm.ppf(1-alpha/2)
    p_sample = sample_data.count(rating)/len(sample_data)
    z_value = (p_sample - null_hypothesis_prop)/math.pow(null_hypothesis_prop*(1-null_hypothesis_prop)/len(sample_data),0.5)
    
    print("sample proportion: %.2f"%p_sample)
    print("alpha: ", alpha)
    print("z-statistic, z-critical: %.2f, %.2f"%(z_value, z_critical))
    if abs(z_value) >= z_critical:
        print("| %.2f | > %.2f"%(z_value, z_critical))
        print(colored("one sample proportion test at alpha %.2f:\t Hypothesis rejected | H0: p = %.2f"%(alpha, null_hypothesis_prop), 'red'))
    else:
        print("|%.2f| < %.2f"%(z_value, z_critical))
        print(colored("one sample proportion test at alpha %.2f:\t Hypothesis accepted | H0: p = %.2f"%(alpha, null_hypothesis_prop), 'green'))

def proportion_test_two_sample(sample_data1, sample_data2, rating, alpha):
    n1 = len(sample_data1)
    n2 = len(sample_data2)
    p1 = sample_data1.count(rating)/len(sample_data1)
    p2 = sample_data2.count(rating)/len(sample_data2)
    p = (n1*p1 + n2*p2)/(n1+n2)
    q = 1-p
    check1, check2, check3, check4 = n1*p, n1*q, n2*p, n2*q
    print("Sample size 1: ", len(sample_data1))
    print("Sample size 2: ", len(sample_data2))
    print("n1p, n1q, n2p, n2q = %.2f, %.2f, %.2f, %.2f\n"%(check1, check2, check3, check4))
    if(check1 < 5 or check2 < 5 or check3 < 5 or check4 < 5):
        print("Check failed. Can't perform proportion test")
        return
    z_critical = norm.ppf(1-alpha/2)
    SEM = math.pow(p*q*(1/n1 + 1/n2), 0.5)
    z_value = (p1 - p2)/SEM
    print("Sample proportions: %.2f, %.2f"%(p1, p2))
    print("alpha: ", alpha)
    print("z-statistic, z-critical: %.2f, %.2f"%(z_value, z_critical))
    if abs(z_value) > z_critical:
        print(colored("two sample proportion test at alpha %.2f:\t Hypothesis rejected | H0: p1 >= p2"%alpha, 'red'))
    else:
        print(colored("two sample proportion test at alpha %.2f:\t Hypothesis accepted | H0: p1 < p2"%alpha, 'green'))

def runs_test(sample_data, alpha):
    median_value = np.median(sample_data)
    binary_data = [1 if x > median_value else 0 for x in sample_data if x != median_value]
    print("Median: ",median_value)
    print("Runs data: ")
    for x in binary_data:
        if x == 1:
            print('A, ',end='')
        else:
            print('B, ',end='')
    runs_count = 0
    current_run = None

    for bit in binary_data:
        if bit != current_run:
            runs_count += 1
            current_run = bit

    print("\nNumber of runs: ", runs_count)
    n1 = binary_data.count(1)
    n2 = binary_data.count(0)
    print("A: ", n1)
    print("B: ", n2)
    critical_value = table[n1-2][n2-2]
    print("alpha: ", alpha)
    print("Critical range: ", critical_value)
    result = critical_value[0] <= runs_count <= critical_value[1]
    print("Result at alpha = %.2f:"%alpha, colored("The sample is not random",'red') if not result else colored("The sample is random",'green'))

def sign_test_one_sample(sample_data, median_value, alpha):

    sampled_data = ['+' if x > median_value else '-' if x < median_value else '0' for x in sample_data]
    print(sampled_data)

    x = sampled_data.count('+') + sampled_data.count('-')
    n = len(sample_data)
    print("Sample size: ", n)
    z = (x+0.05-n/2)/math.pow(n/2,0.5)
    z_critical = norm.ppf(1 - alpha/2)
    print("alpha: ", alpha)
    print("z-statistic, z-critical: %.2f, %.2f"%(z, z_critical))

    if z_critical <= abs(z):
        print(colored("one sample sign test at alpha %.2f:\t Hypothesis rejected | H0: median=%.1f"%(alpha, median_value), 'red'))
    else:
        print(colored("one sample sign test at alpha %.2f:\t Hypothesis accepted | H0: median=%.1f"%(alpha, median_value), 'green'))

def sign_test_paired_sample(dataframe, alpha):
    df = dataframe.copy(deep = True)
    df['diff'] = np.where(df['movie1'] > df['movie2'], '+', np.where(df['movie1'] < df['movie2'], '-', '0'))
    x = min(df['diff'].value_counts().get('+', 0), df['diff'].value_counts().get('-', 0))
    print(df.to_string(index=False))
    df = df.drop('diff', axis=1)

    print("Sample size: ", len(df))
    if alpha == 0.01:
        crit = 5
    elif alpha == 0.05 or alpha == 0.02:
        crit = 6
    else:
        crit = 7
    print("alpha: ", alpha)
    print("critical value:",crit)
    print("calculated score:", x)
    if x <= crit:
        print(colored("paired sign test at alpha %.2f:\t\t Hypothesis rejected | H0: mean difference=%.1f"%(alpha, 0), 'red'))
    else:
        print(colored("paired sign test at alpha %.2f:\t\t Hypothesis accepted | H0: mean difference=%.1f"%(alpha, 0), 'green'))
    

def wilcoxon_ran_sum_test(sample_data1, sample_data2, alpha):
    z_critical = norm.ppf(1 - alpha/2)
    df1 = pd.DataFrame({'rating': sample_data1})
    df1['group'] = 'A'

    df2 = pd.DataFrame({'rating': sample_data2})
    df2['group'] = 'B'
    concatenated_df = pd.concat([df1, df2], ignore_index=True)
    sorted_df = concatenated_df.sort_values(by='rating')

    sorted_df['rank'] = sorted_df['rating'].rank()
    newdf = sorted_df.transpose()
    print(newdf.to_string(index=True, header=False), '\n')
    if(len(sample_data1) < len(sample_data2)):
        R = sorted_df[sorted_df['group'] == 'A']['rank'].sum()
        n1 = len(sample_data1)
        n2 = len(sample_data2)
    else:
        R = sorted_df[sorted_df['group'] == 'A']['rank'].sum()
        n1 = len(sample_data1)
        n2 = len(sample_data2)
    mean = n1*(n1+n2+1)/2
    var = math.pow(n1*n2*(n1+n2+1)/12, 0.5)
    print("Rank Sum = ", R)
    print("n1 = ", n1)
    print("n2 = ", n2)
    z_value = (R - mean)/var
    print("alpha: ", alpha)
    print("z-statistic, z-critical: %.2f, %.2f"%(z_value, z_critical))
    if abs(z_value) > z_critical:
        print(colored("Wilcoxon rank sum test at alpha %.2f: Hypothesis rejected | H0: mean1 = mean2"%alpha, 'red'))
    else:
        print(colored("Wilcoxon rank sum test at alpha %.2f: Hypothesis accepted | H0: mean1 = mean2"%alpha, 'green'))

def wilcoxon_signed_rank_test(paired_df, alpha):
    df = paired_df.copy(deep=True)
    print("Sample size: ", len(df))
    df['diff'] = df['movie1'] - df['movie2']
    df['abs_diff'] = abs(df['diff'])
    df = df[df['abs_diff'] != 0]
    df['rank'] = df['abs_diff'].rank()
    df['signed_rank'] = df.apply(lambda row: 0 if row['diff'] == 0 else row['rank'] * (-1 if row['diff'] < 0 else 1), axis=1)
    print(df.to_string(index=False),"\n")

    positive_rank_sum = df[df['signed_rank'] > 0]['signed_rank'].sum()
    negative_rank_sum = abs(df[df['signed_rank'] < 0]['signed_rank'].sum())
    Ws = min(positive_rank_sum, negative_rank_sum)
    n=len(df)
    SEM = math.pow(n*(n+1)*(2*n+1)/24,0.5)
    z_value = (Ws - (n*(n+1)/4))/SEM
    z_critical = norm.ppf(1-alpha/2)
    print("z-statistic, z-critical: %.2f, %.2f"%(z_value, z_critical))
    if abs(z_value) > z_critical:
        print(colored("Wilcoxon signed rank test at alpha %.2f: Hypothesis rejected | H0: mean1 = mean2"%alpha, 'red'))
    else:
        print(colored("Wilcoxon signed rank test at alpha %.2f: Hypothesis accepted | H0: mean1 = mean2"%alpha, 'green'))
    print(Ws)

def chi_square_good_fit_test(df, random_movies1, alpha):
    selected_movies = random.sample(random_movies1, 5)
    filtered_df1_each = df[df['movieid'].isin(selected_movies)]
    
    def sample_ratings(group):
        return group.sample(n=10, random_state=42)

    dataset1_each = filtered_df1_each.groupby('movieid', group_keys=False).apply(sample_ratings)

    dataset1_each.loc[:, 'Category'] = dataset1_each['rating'].apply(lambda x: 'Good' if x >= 3 else 'Bad')

    observed_df = pd.crosstab(index=dataset1_each['Category'], columns=dataset1_each['movieid'])
    
    print("-----------------------Contingency Table----------------------")
    print(observed_df)

    observed_values = observed_df.iloc[1].values

    total_count = np.sum(observed_values)
    expected_values = [total_count / len(observed_values)] * len(observed_values)

        # Calculate the chi-square statistic
    chi2_stat = np.sum((observed_values - expected_values)**2 / expected_values)

    # Calculate the degrees of freedom
    dof = len(observed_values) - 1
    print("degrees of freedom:",dof)

    # Find the critical value from the chi-square distribution table
    critical_value = 14.860
    print("\n")
    print("dof: ", dof)
    print("alpha: ", 0.05)
    print("Chi-square Statistic:", chi2_stat)
    print("Critical Value:", critical_value)
    print("\n")

    if chi2_stat > critical_value:
        print("Reject the null hypothesis. There is a preference by users for particular movies.")
    else:
        print("Fail to reject the null hypothesis. There is not enough evidence of preference by users for particular movies.")

def chi_square_independence_test(df, random_movies1, alpha):
    selected_movies = random.sample(random_movies1, 2)
    filtered_df1_each = df[df['movieid'].isin(selected_movies)]

    def sample_ratings(group):
        return group.sample(n=10, random_state=42)

    dataset1_each = filtered_df1_each.groupby('movieid', group_keys=False).apply(sample_ratings)
    dataset1_each.loc[:, 'Category'] = dataset1_each['rating'].apply(lambda x: 'Good' if x >= 3 else 'Bad')

    observed_df = pd.crosstab(index=dataset1_each['Category'], columns=dataset1_each['movieid'])
    print("-----------------------Observed Table----------------------\n")
    print(observed_df)

    row_totals = np.sum(observed_df.values, axis=1)
    column_totals = np.sum(observed_df.values, axis=0)
    total = np.sum(observed_df.values)

    #used for calculating values
    expected_values = np.outer(row_totals, column_totals) / total

    #used for display purpose only with index
    expected_df = pd.DataFrame(expected_values, index=observed_df.index, columns=observed_df.columns)
    print("-----------------------Expected Table----------------------\n")
    print(expected_df)

    # Calculate the test value (chi-square statistic)
    chi2_stat = np.sum((observed_df.values - expected_values)**2 / expected_values)

    # Degrees of freedom
    df = (len(row_totals) - 1) * (len(column_totals) - 1)
    print("Degrees of freedom:",df,"\n")

    # Find the critical value from the chi-square distribution table
    critical_value = chi2.ppf(1 - alpha, df)

    print("Chi-square Statistic:", chi2_stat)
    print("Critical Value:", critical_value)

    # Step 4: Make the decision.
    if chi2_stat > critical_value:
        print("Reject the null hypothesis. There is a no association on movie being good and released on a particular year")
    else:
        print("Fail to reject the null hypothesis. There is association by users and movie being good that is released on particular day")



def chi_square_homogenity_test(df, random_movies1, alpha):
    selected_movies = random.sample(random_movies1, 3)
    filtered_df1_each = df[df['movieid'].isin(selected_movies)]

    def sample_ratings(group):
        return group.sample(n=10, random_state=42)

    dataset1_each = filtered_df1_each.groupby('movieid', group_keys=False).apply(sample_ratings)
    dataset1_each.loc[:, 'Category'] = dataset1_each['rating'].apply(lambda x: 'Good' if x >= 3 else 'Bad')

    observed_df = pd.crosstab(index=dataset1_each['Category'], columns=dataset1_each['movieid'])
    print("-----------------------Observed Table----------------------\n")
    print(observed_df)

    row_totals = np.sum(observed_df.values, axis=1)
    column_totals = np.sum(observed_df.values, axis=0)
    total = np.sum(observed_df.values)

    #used for calculating values
    expected_values = np.outer(row_totals, column_totals) / total

    #used for display purpose only with index
    expected_df = pd.DataFrame(expected_values, index=observed_df.index, columns=observed_df.columns)
    print("-----------------------Expected Table----------------------\n")
    print(expected_df)

    # Calculate the test value (chi-square statistic)
    chi2_stat = np.sum((observed_df.values - expected_values)**2 / expected_values)

    # Degrees of freedom
    df = (len(row_totals) - 1) * (len(column_totals) - 1)
    print("Degrees of freedom:",df,"\n")

    # Find the critical value from the chi-square distribution table
    critical_value = chi2.ppf(1 - alpha, df)

    print("Chi-square Statistic:", chi2_stat)
    print("Critical Value:", critical_value)

    # Step 4: Make the decision.
    if chi2_stat > critical_value:
        print("Reject the null hypothesis.Atleast one mean differs from the other")
    else:
        print("Fail to reject the null hypothesis.The proportion of good and bad ratings for all 3 movies released in a particular is same")


def two_by_two_contingency_test(df, random_movies1, alpha):
    selected_movies = random.sample(random_movies1, 2)
    filtered_df1_each = df[df['movieid'].isin(selected_movies)]

    def sample_ratings(group):
        return group.sample(n=10, random_state=42)

    dataset1_each = filtered_df1_each.groupby('movieid', group_keys=False).apply(sample_ratings)
    dataset1_each.loc[:, 'Category'] = dataset1_each['rating'].apply(lambda x: 'Good' if x >= 3 else 'Bad')

    observed_df = pd.crosstab(index=dataset1_each['Category'], columns=dataset1_each['movieid'])
    print("-----------------------Observed Table----------------------\n")
    print(observed_df)        

    a = observed_df.iloc[0, 0]  # Count for Good ratings, Movie 1
    b = observed_df.iloc[0, 1]  # Count for Good ratings, Movie 2
    c = observed_df.iloc[1, 0]  # Count for Bad ratings, Movie 1
    d = observed_df.iloc[1, 1]  # Count for Bad ratings, Movie 2

    df = 1  # Degrees of freedom for a 2x2 table
    critical_value = chi2.ppf(1 - alpha, df)

    #n
    N = np.sum(observed_df.values)

    row_totals = np.sum(observed_df.values, axis=1)
    column_totals = np.sum(observed_df.values, axis=0)
    total = np.sum(observed_df.values)

    #used for calculating values
    expected_values = np.outer(row_totals, column_totals) / total
 

    for i in range(expected_values.shape[0]):
        for j in range(expected_values.shape[1]):
            if(expected_values[i][j] < 5):
                print("There is a value less than 5 so Yates Correction test need to be applied")
                return

    # Step 3: Compute the test value using the formula.
    chi2_stat = N * (a*d - b*c)**2 / ((a + b)*(c + d)*(a + c)*(b + d))

    print("Chi-square Statistic:", chi2_stat)
    print("Critical Value:", critical_value)
    
    # Step 4: Make the decision.
    if chi2_stat > critical_value:
        print("Reject the null hypothesis. There is an association between the two variables.")
    else:
        print("Fail to reject the null hypothesis. There is no association between the two variables.")


def yates_correction_test(df, random_movies1, alpha):
    selected_movies = random.sample(random_movies1, 2)
    filtered_df1_each = df[df['movieid'].isin(selected_movies)]

    def sample_ratings(group):
        return group.sample(n=10, random_state=42)

    dataset1_each = filtered_df1_each.groupby('movieid', group_keys=False).apply(sample_ratings)
    dataset1_each.loc[:, 'Category'] = dataset1_each['rating'].apply(lambda x: 'Good' if x >= 3 else 'Bad')

    observed_df = pd.crosstab(index=dataset1_each['Category'], columns=dataset1_each['movieid'])
    print("-----------------------Observed Table----------------------\n")
    print(observed_df)        

    a = observed_df.iloc[0, 0]  # Count for Good ratings, Movie 1
    b = observed_df.iloc[0, 1]  # Count for Good ratings, Movie 2
    c = observed_df.iloc[1, 0]  # Count for Bad ratings, Movie 1
    d = observed_df.iloc[1, 1]  # Count for Bad ratings, Movie 2

    row_totals = np.sum(observed_df.values, axis=1)
    column_totals = np.sum(observed_df.values, axis=0)
    total = np.sum(observed_df.values)

    #used for calculating values
    expected_values = np.outer(row_totals, column_totals) / total
 
    check = 1

    for i in range(expected_values.shape[0]):
        for j in range(expected_values.shape[1]):
            if(expected_values[i][j] < 5):
                check = 0
                continue


    if(check):
        print("Yates test conditions aren't satisfied i.e there is not even atleast one value less than 5")
        return

    df = 1  # Degrees of freedom for a 2x2 table
    critical_value = chi2.ppf(1 - alpha, df)

    #n
    N = np.sum(observed_df.values)

    # Step 3: Compute the test value using the formula.
    chi2_stat = N * (abs(a*d - b*c)-(N/2))**2 / ((a + b)*(c + d)*(a + c)*(b + d))

    print("Chi-square Statistic:", chi2_stat)
    print("Critical Value:", critical_value)
    
    # Step 4: Make the decision.
    if chi2_stat > critical_value:
        print("Reject the null hypothesis. There is an association between the two variables.")
    else:
        print("Fail to reject the null hypothesis. There is no association between the two variables.")        