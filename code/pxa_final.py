# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind, shapiro, chi2_contingency, fisher_exact
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# %%
# Identify project's root directory
project_path = os.path.dirname(os.getcwd())

# Data import: read clinical data as pandas dataframe
pxa_path = os.path.join(project_path, 'data' ,'pxa_data.csv')
pxa = pd.read_csv(pxa_path)
pxa.set_index('ID', inplace=True)

# %%
# Create the two groups based on the "tumor-epilepsy" column
pxa_initial_sz = pxa[pxa['first_tumor_sz'] == 1]
pxa_initial_other = pxa[pxa['first_tumor_sz'] == 0]
pxa_tre = pxa[pxa['tumor_epilepsy'] == 1]
pxa_no_tre = pxa[pxa['tumor_epilepsy'] == 0]

# %% [markdown]
# ## Table 1: Clinical characteristics of seizures vs. no seizures

# %% [markdown]
# ### Totals

# %%
# Calculate the total number of patients in each group
total_initial_sz = len(pxa_initial_sz)
total_initial_other = len(pxa_initial_other)
total_tre = len(pxa_tre)
total_no_tre = len(pxa_no_tre)

# Calculate the total number of patients
total_patients = total_initial_sz + total_initial_other

# Calculate the proportion of total for each group
prop_initial_sz = total_initial_sz / total_patients
prop_initial_other = total_initial_other / total_patients
prop_tre = total_tre / total_patients
prop_no_tre = total_no_tre / total_patients

print(f"Total number of patients in pxa_initial_sz: {total_initial_sz}")
print(f"Proportion of total patients in pxa_initial_sz: {prop_initial_sz:.2f}")

print(f"Total number of patients in pxa_initial_other: {total_initial_other}")
print(f"Proportion of total patients in pxa_initial_other: {prop_initial_other:.2f}")

print(f"Total number of patients in pxa_tre: {total_tre}")
print(f"Proportion of total patients in pxa_tre: {prop_tre:.2f}")

print(f"Total number of patients in pxa_no_tre: {total_no_tre}")
print(f"Proportion of total patients in pxa_no_tre: {prop_no_tre:.2f}")

# %% [markdown]
# ### Continuous data: age of onset, tumor size

# %%
# Calculate the median age and range for each group
def calculate_median_and_range(df, column):
    median = df[column].median()
    min_val = df[column].min()
    max_val = df[column].max()
    return median, min_val, max_val

median_age_initial_sz, min_age_initial_sz, max_age_initial_sz = calculate_median_and_range(pxa_initial_sz, 'presentation_1_age')
median_age_initial_other, min_age_initial_other, max_age_initial_other = calculate_median_and_range(pxa_initial_other, 'presentation_1_age')
median_age_tre, min_age_tre, max_age_tre = calculate_median_and_range(pxa_tre, 'presentation_1_age')
median_age_no_tre, min_age_no_tre, max_age_no_tre = calculate_median_and_range(pxa_no_tre, 'presentation_1_age')

print(f"Median age of initial presentation in pxa_initial_sz: {median_age_initial_sz}")
print(f"Range of age of initial presentation in pxa_initial_sz: {min_age_initial_sz} - {max_age_initial_sz}")

print(f"Median age of initial presentation in pxa_initial_other: {median_age_initial_other}")
print(f"Range of age of initial presentation in pxa_initial_other: {min_age_initial_other} - {max_age_initial_other}")

print(f"Median age of initial presentation in pxa_tre: {median_age_tre}")
print(f"Range of age of initial presentation in pxa_tre: {min_age_tre} - {max_age_tre}")

print(f"Median age of initial presentation in pxa_no_tre: {median_age_no_tre}")
print(f"Range of age of initial presentation in pxa_no_tre: {min_age_no_tre} - {max_age_no_tre}")

# Check for normality using the Shapiro-Wilk test
def check_normality(df, column):
    return shapiro(df[column])

shapiro_sz = check_normality(pxa_initial_sz, 'presentation_1_age')
shapiro_other = check_normality(pxa_initial_other, 'presentation_1_age')
shapiro_tre = check_normality(pxa_tre, 'presentation_1_age')
shapiro_no_tre = check_normality(pxa_no_tre, 'presentation_1_age')

print(f"Shapiro-Wilk test for pxa_initial_sz: {shapiro_sz}")
print(f"Shapiro-Wilk test for pxa_initial_other: {shapiro_other}")
print(f"Shapiro-Wilk test for pxa_tre: {shapiro_tre}")
print(f"Shapiro-Wilk test for pxa_no_tre: {shapiro_no_tre}")

# Perform the appropriate test based on normality
def perform_statistical_test(df1, df2, column):
    shapiro1 = check_normality(df1, column)
    shapiro2 = check_normality(df2, column)
    if shapiro1.pvalue > 0.05 and shapiro2.pvalue > 0.05:
        t_statistic, p_value = ttest_ind(df1[column], df2[column])
        test_used = "t-test"
    else:
        u_statistic, p_value = mannwhitneyu(df1[column], df2[column])
        test_used = "Mann-Whitney U test"
    return test_used, t_statistic if test_used == "t-test" else u_statistic, p_value

test_used_age, statistic_age, p_value_age = perform_statistical_test(pxa_initial_sz, pxa_initial_other, 'presentation_1_age')
print(f"{test_used_age} for age of initial presentation: statistic={statistic_age}, p-value={p_value_age}")

test_used_age_tre, statistic_age_tre, p_value_age_tre = perform_statistical_test(pxa_tre, pxa_no_tre, 'presentation_1_age')
print(f"{test_used_age_tre} for age of initial presentation: statistic={statistic_age_tre}, p-value={p_value_age_tre}")

# Ensure tumor_size is numeric, converting non-numeric values to NaN and then dropping them
def filter_numeric(df, column):
    df_filtered = df.copy()
    df_filtered[column] = pd.to_numeric(df_filtered[column], errors='coerce')
    return df_filtered.dropna(subset=[column])

# Filter and analyze tumor_size
pxa_initial_sz_filtered = filter_numeric(pxa_initial_sz, 'tumor_size')
pxa_initial_other_filtered = filter_numeric(pxa_initial_other, 'tumor_size')
pxa_tre_filtered = filter_numeric(pxa_tre, 'tumor_size')
pxa_no_tre_filtered = filter_numeric(pxa_no_tre, 'tumor_size')

# Calculate the mean tumor size and standard deviation for each group
def calculate_mean_and_std(df, column):
    mean = df[column].mean()
    std = df[column].std()
    return mean, std

mean_tumor_size_initial_sz, std_tumor_size_initial_sz = calculate_mean_and_std(pxa_initial_sz_filtered, 'tumor_size')
mean_tumor_size_initial_other, std_tumor_size_initial_other = calculate_mean_and_std(pxa_initial_other_filtered, 'tumor_size')
mean_tumor_size_tre, std_tumor_size_tre = calculate_mean_and_std(pxa_tre_filtered, 'tumor_size')
mean_tumor_size_no_tre, std_tumor_size_no_tre = calculate_mean_and_std(pxa_no_tre_filtered, 'tumor_size')

print(f"Mean tumor size in pxa_initial_sz: {mean_tumor_size_initial_sz}")
print(f"Standard deviation of tumor size in pxa_initial_sz: {std_tumor_size_initial_sz}")

print(f"Mean tumor size in pxa_initial_other: {mean_tumor_size_initial_other}")
print(f"Standard deviation of tumor size in pxa_initial_other: {std_tumor_size_initial_other}")

print(f"Mean tumor size in pxa_tre: {mean_tumor_size_tre}")
print(f"Standard deviation of tumor size in pxa_tre: {std_tumor_size_tre}")

print(f"Mean tumor size in pxa_no_tre: {mean_tumor_size_no_tre}")
print(f"Standard deviation of tumor size in pxa_no_tre: {std_tumor_size_no_tre}")

# Check for normality using the Shapiro-Wilk test
shapiro_tumor_size_sz = check_normality(pxa_initial_sz_filtered, 'tumor_size')
shapiro_tumor_size_other = check_normality(pxa_initial_other_filtered, 'tumor_size')
shapiro_tumor_size_tre = check_normality(pxa_tre_filtered, 'tumor_size')
shapiro_tumor_size_no_tre = check_normality(pxa_no_tre_filtered, 'tumor_size')

print(f"Shapiro-Wilk test for tumor size in pxa_initial_sz: {shapiro_tumor_size_sz}")
print(f"Shapiro-Wilk test for tumor size in pxa_initial_other: {shapiro_tumor_size_other}")
print(f"Shapiro-Wilk test for tumor size in pxa_tre: {shapiro_tumor_size_tre}")
print(f"Shapiro-Wilk test for tumor size in pxa_no_tre: {shapiro_tumor_size_no_tre}")

# Perform the appropriate test based on normality
test_used_tumor_size, statistic_tumor_size, p_value_tumor_size = perform_statistical_test(pxa_initial_sz_filtered, pxa_initial_other_filtered, 'tumor_size')
print(f"{test_used_tumor_size} for tumor size: statistic={statistic_tumor_size}, p-value={p_value_tumor_size}")

test_used_tumor_size_tre, statistic_tumor_size_tre, p_value_tumor_size_tre = perform_statistical_test(pxa_tre_filtered, pxa_no_tre_filtered, 'tumor_size')
print(f"{test_used_tumor_size_tre} for tumor size: statistic={statistic_tumor_size_tre}, p-value={p_value_tumor_size_tre}")

# Ensure tumor_mutational_burden is numeric, converting non-numeric values to NaN and then dropping them
pxa_initial_sz_filtered = filter_numeric(pxa_initial_sz, 'tumor_mutational_burden')
pxa_initial_other_filtered = filter_numeric(pxa_initial_other, 'tumor_mutational_burden')
pxa_tre_filtered = filter_numeric(pxa_tre, 'tumor_mutational_burden')
pxa_no_tre_filtered = filter_numeric(pxa_no_tre, 'tumor_mutational_burden')

# Calculate the mean tumor mutational burden and standard deviation for each group
mean_tumor_mutational_burden_initial_sz, std_tumor_mutational_burden_initial_sz = calculate_mean_and_std(pxa_initial_sz_filtered, 'tumor_mutational_burden')
mean_tumor_mutational_burden_initial_other, std_tumor_mutational_burden_initial_other = calculate_mean_and_std(pxa_initial_other_filtered, 'tumor_mutational_burden')
mean_tumor_mutational_burden_tre, std_tumor_mutational_burden_tre = calculate_mean_and_std(pxa_tre_filtered, 'tumor_mutational_burden')
mean_tumor_mutational_burden_no_tre, std_tumor_mutational_burden_no_tre = calculate_mean_and_std(pxa_no_tre_filtered, 'tumor_mutational_burden')

print(f"Mean tumor mutational burden in pxa_initial_sz: {mean_tumor_mutational_burden_initial_sz}")
print(f"Standard deviation of tumor mutational burden in pxa_initial_sz: {std_tumor_mutational_burden_initial_sz}")

print(f"Mean tumor mutational burden in pxa_initial_other: {mean_tumor_mutational_burden_initial_other}")
print(f"Standard deviation of tumor mutational burden in pxa_initial_other: {std_tumor_mutational_burden_initial_other}")

print(f"Mean tumor mutational burden in pxa_tre: {mean_tumor_mutational_burden_tre}")
print(f"Standard deviation of tumor mutational burden in pxa_tre: {std_tumor_mutational_burden_tre}")

print(f"Mean tumor mutational burden in pxa_no_tre: {mean_tumor_mutational_burden_no_tre}")
print(f"Standard deviation of tumor mutational burden in pxa_no_tre: {std_tumor_mutational_burden_no_tre}")

# Check for normality using the Shapiro-Wilk test
shapiro_tumor_mutational_burden_sz = check_normality(pxa_initial_sz_filtered, 'tumor_mutational_burden')
shapiro_tumor_mutational_burden_other = check_normality(pxa_initial_other_filtered, 'tumor_mutational_burden')
shapiro_tumor_mutational_burden_tre = check_normality(pxa_tre_filtered, 'tumor_mutational_burden')
shapiro_tumor_mutational_burden_no_tre = check_normality(pxa_no_tre_filtered, 'tumor_mutational_burden')

print(f"Shapiro-Wilk test for tumor mutational burden in pxa_initial_sz: {shapiro_tumor_mutational_burden_sz}")
print(f"Shapiro-Wilk test for tumor mutational burden in pxa_initial_other: {shapiro_tumor_mutational_burden_other}")
print(f"Shapiro-Wilk test for tumor mutational burden in pxa_tre: {shapiro_tumor_mutational_burden_tre}")
print(f"Shapiro-Wilk test for tumor mutational burden in pxa_no_tre: {shapiro_tumor_mutational_burden_no_tre}")

# Perform the appropriate test based on normality
test_used_tumor_mutational_burden, statistic_tumor_mutational_burden, p_value_tumor_mutational_burden = perform_statistical_test(pxa_initial_sz_filtered, pxa_initial_other_filtered, 'tumor_mutational_burden')
print(f"{test_used_tumor_mutational_burden} for tumor mutational burden: statistic={statistic_tumor_mutational_burden}, p-value={p_value_tumor_mutational_burden}")

test_used_tumor_mutational_burden_tre, statistic_tumor_mutational_burden_tre, p_value_tumor_mutational_burden_tre = perform_statistical_test(pxa_tre_filtered, pxa_no_tre_filtered, 'tumor_mutational_burden')
print(f"{test_used_tumor_mutational_burden_tre} for tumor mutational burden: statistic={statistic_tumor_mutational_burden_tre}, p-value={p_value_tumor_mutational_burden_tre}")

# %% [markdown]
# ### Categorical data: female/male, tumor grade, tumor laterality

# %%
def perform_test(group1, group2, column_name):
    # Ensure the column is numeric, converting non-numeric values to NaN and then dropping them
    group1_filtered = group1.copy()
    group1_filtered[column_name] = pd.to_numeric(group1_filtered[column_name], errors='coerce')
    group1_filtered = group1_filtered.dropna(subset=[column_name])

    group2_filtered = group2.copy()
    group2_filtered[column_name] = pd.to_numeric(group2_filtered[column_name], errors='coerce')
    group2_filtered = group2_filtered.dropna(subset=[column_name])

    # Calculate the total numbers of 1s and 0s in each group
    total_1_group1 = (group1_filtered[column_name] == 1).sum()
    total_0_group1 = (group1_filtered[column_name] == 0).sum()

    total_1_group2 = (group2_filtered[column_name] == 1).sum()
    total_0_group2 = (group2_filtered[column_name] == 0).sum()

    # Calculate the proportions
    proportion_1_group1 = total_1_group1 / (total_1_group1 + total_0_group1)
    proportion_1_group2 = total_1_group2 / (total_1_group2 + total_0_group2)

    print(f"Total number of 1s in {column_name} for group1: {total_1_group1}")
    print(f"Total number of 0s in {column_name} for group1: {total_0_group1}")
    print(f"Proportion of 1s in {column_name} for group1: {proportion_1_group1:.2f}")

    print(f"Total number of 1s in {column_name} for group2: {total_1_group2}")
    print(f"Total number of 0s in {column_name} for group2: {total_0_group2}")
    print(f"Proportion of 1s in {column_name} for group2: {proportion_1_group2:.2f}")

    # Create the contingency table
    contingency_table = [
        [total_1_group1, total_0_group1],
        [total_1_group2, total_0_group2]
    ]

    # Perform the Chi-Square test or Fisher's Exact Test
    if min(total_1_group1, total_0_group1, total_1_group2, total_0_group2) < 5:
        # Use Fisher's Exact Test if any expected frequency is less than 5
        _, p_value = fisher_exact(contingency_table)
        test_used = "Fisher's Exact Test"
    else:
        # Use Chi-Square test otherwise
        _, p_value, _, _ = chi2_contingency(contingency_table)
        test_used = "Chi-Square test"

    print(f"{test_used} for {column_name}: p-value={p_value}")

# Perform the tests for sex_female
print("Comparing pxa_initial_sz vs. pxa_initial_other for sex_female")
perform_test(pxa_initial_sz, pxa_initial_other, 'sex_female')

print("Comparing pxa_tre vs. pxa_no_tre for sex_female")
perform_test(pxa_tre, pxa_no_tre, 'sex_female')

# Ensure tumor_grade_first is numeric, converting non-numeric values to NaN and then dropping them
def filter_numeric(df, column):
    df_filtered = df.copy()
    df_filtered[column] = pd.to_numeric(df_filtered[column], errors='coerce')
    return df_filtered.dropna(subset=[column])

pxa_initial_sz_filtered = filter_numeric(pxa_initial_sz, 'tumor_grade_first')
pxa_initial_other_filtered = filter_numeric(pxa_initial_other, 'tumor_grade_first')
pxa_tre_filtered = filter_numeric(pxa_tre, 'tumor_grade_first')
pxa_no_tre_filtered = filter_numeric(pxa_no_tre, 'tumor_grade_first')

# Calculate the total numbers of grade 2 and grade 3 tumors in each group
def calculate_grades(df, column):
    total_grade2 = (df[column] == 2).sum()
    total_grade3 = (df[column] == 3).sum()
    return total_grade2, total_grade3

total_grade2_initial_sz, total_grade3_initial_sz = calculate_grades(pxa_initial_sz_filtered, 'tumor_grade_first')
total_grade2_initial_other, total_grade3_initial_other = calculate_grades(pxa_initial_other_filtered, 'tumor_grade_first')
total_grade2_tre, total_grade3_tre = calculate_grades(pxa_tre_filtered, 'tumor_grade_first')
total_grade2_no_tre, total_grade3_no_tre = calculate_grades(pxa_no_tre_filtered, 'tumor_grade_first')

print(f"Total number of grade 2 tumors in pxa_initial_sz: {total_grade2_initial_sz}")
print(f"Total number of grade 3 tumors in pxa_initial_sz: {total_grade3_initial_sz}")

print(f"Total number of grade 2 tumors in pxa_initial_other: {total_grade2_initial_other}")
print(f"Total number of grade 3 tumors in pxa_initial_other: {total_grade3_initial_other}")

print(f"Total number of grade 2 tumors in pxa_tre: {total_grade2_tre}")
print(f"Total number of grade 3 tumors in pxa_tre: {total_grade3_tre}")

print(f"Total number of grade 2 tumors in pxa_no_tre: {total_grade2_no_tre}")
print(f"Total number of grade 3 tumors in pxa_no_tre: {total_grade3_no_tre}")

# Perform the Chi-Square test or Fisher's Exact Test for tumor grades
def perform_grade_test(total_grade2_group1, total_grade3_group1, total_grade2_group2, total_grade3_group2):
    contingency_table_grades = [
        [total_grade2_group1, total_grade3_group1],
        [total_grade2_group2, total_grade3_group2]
    ]
    if min(total_grade2_group1, total_grade3_group1, total_grade2_group2, total_grade3_group2) < 5:
        # Use Fisher's Exact Test if any expected frequency is less than 5
        _, p_value_grades = fisher_exact(contingency_table_grades)
        test_used_grades = "Fisher's Exact Test"
    else:
        # Use Chi-Square test otherwise
        _, p_value_grades, _, _ = chi2_contingency(contingency_table_grades)
        test_used_grades = "Chi-Square test"
    return test_used_grades, p_value_grades

test_used_grades, p_value_grades = perform_grade_test(total_grade2_initial_sz, total_grade3_initial_sz, total_grade2_initial_other, total_grade3_initial_other)
print(f"{test_used_grades} for tumor grades in pxa_initial_sz vs. pxa_initial_other: p-value={p_value_grades}")

test_used_grades_tre, p_value_grades_tre = perform_grade_test(total_grade2_tre, total_grade3_tre, total_grade2_no_tre, total_grade3_no_tre)
print(f"{test_used_grades_tre} for tumor grades in pxa_tre vs. pxa_no_tre: p-value={p_value_grades_tre}")

# Ensure tumor_laterality is either 'L' or 'R'
pxa_initial_sz_filtered = pxa_initial_sz[pxa_initial_sz['tumor_laterality'].isin(['L', 'R'])]
pxa_initial_other_filtered = pxa_initial_other[pxa_initial_other['tumor_laterality'].isin(['L', 'R'])]
pxa_tre_filtered = pxa_tre[pxa_tre['tumor_laterality'].isin(['L', 'R'])]
pxa_no_tre_filtered = pxa_no_tre[pxa_no_tre['tumor_laterality'].isin(['L', 'R'])]

# Calculate the total numbers of left and right tumors in each group
def calculate_laterality(df, column):
    total_left = (df[column] == 'L').sum()
    total_right = (df[column] == 'R').sum()
    return total_left, total_right

total_left_initial_sz, total_right_initial_sz = calculate_laterality(pxa_initial_sz_filtered, 'tumor_laterality')
total_left_initial_other, total_right_initial_other = calculate_laterality(pxa_initial_other_filtered, 'tumor_laterality')
total_left_tre, total_right_tre = calculate_laterality(pxa_tre_filtered, 'tumor_laterality')
total_left_no_tre, total_right_no_tre = calculate_laterality(pxa_no_tre_filtered, 'tumor_laterality')

print(f"Total number of left tumors in pxa_initial_sz: {total_left_initial_sz}")
print(f"Total number of right tumors in pxa_initial_sz: {total_right_initial_sz}")

print(f"Total number of left tumors in pxa_initial_other: {total_left_initial_other}")
print(f"Total number of right tumors in pxa_initial_other: {total_right_initial_other}")

print(f"Total number of left tumors in pxa_tre: {total_left_tre}")
print(f"Total number of right tumors in pxa_tre: {total_right_tre}")

print(f"Total number of left tumors in pxa_no_tre: {total_left_no_tre}")
print(f"Total number of right tumors in pxa_no_tre: {total_right_no_tre}")

# Perform the Chi-Square test or Fisher's Exact Test for tumor laterality
def perform_laterality_test(total_left_group1, total_right_group1, total_left_group2, total_right_group2):
    contingency_table_laterality = [
        [total_left_group1, total_right_group1],
        [total_left_group2, total_right_group2]
    ]
    if min(total_left_group1, total_right_group1, total_left_group2, total_right_group2) < 5:
        # Use Fisher's Exact Test if any expected frequency is less than 5
        _, p_value_laterality = fisher_exact(contingency_table_laterality)
        test_used_laterality = "Fisher's Exact Test"
    else:
        # Use Chi-Square test otherwise
        _, p_value_laterality, _, _ = chi2_contingency(contingency_table_laterality)
        test_used_laterality = "Chi-Square test"
    return test_used_laterality, p_value_laterality

test_used_laterality, p_value_laterality = perform_laterality_test(total_left_initial_sz, total_right_initial_sz, total_left_initial_other, total_right_initial_other)
print(f"{test_used_laterality} for tumor laterality in pxa_initial_sz vs. pxa_initial_other: p-value={p_value_laterality}")

test_used_laterality_tre, p_value_laterality_tre = perform_laterality_test(total_left_tre, total_right_tre, total_left_no_tre, total_right_no_tre)
print(f"{test_used_laterality_tre} for tumor laterality in pxa_tre vs. pxa_no_tre: p-value={p_value_laterality_tre}")

# %% [markdown]
# ### Tumor locations

# %%
def perform_test(group1, group2, column_name):
    # Ensure the column is numeric, converting non-numeric values to NaN and then dropping them
    group1_filtered = group1.copy()
    group1_filtered[column_name] = pd.to_numeric(group1_filtered[column_name], errors='coerce')
    group1_filtered = group1_filtered.dropna(subset=[column_name])

    group2_filtered = group2.copy()
    group2_filtered[column_name] = pd.to_numeric(group2_filtered[column_name], errors='coerce')
    group2_filtered = group2_filtered.dropna(subset=[column_name])

    # Calculate the total numbers of 1s and 0s in each group
    total_1_group1 = (group1_filtered[column_name] == 1).sum()
    total_0_group1 = (group1_filtered[column_name] == 0).sum()

    total_1_group2 = (group2_filtered[column_name] == 1).sum()
    total_0_group2 = (group2_filtered[column_name] == 0).sum()

    # Calculate the proportions
    proportion_1_group1 = total_1_group1 / (total_1_group1 + total_0_group1)
    proportion_1_group2 = total_1_group2 / (total_1_group2 + total_0_group2)

    print(f"Total number of 1s in {column_name} for group1: {total_1_group1}")
    print(f"Proportion of 1s in {column_name} for group1: {proportion_1_group1:.2f}")

    print(f"Total number of 1s in {column_name} for group2: {total_1_group2}")
    print(f"Proportion of 1s in {column_name} for group2: {proportion_1_group2:.2f}")

    # Create the contingency table
    contingency_table = [
        [total_1_group1, total_0_group1],
        [total_1_group2, total_0_group2]
    ]

    # Perform the Chi-Square test or Fisher's Exact Test
    if min(total_1_group1, total_0_group1, total_1_group2, total_0_group2) < 5:
        # Use Fisher's Exact Test if any expected frequency is less than 5
        _, p_value = fisher_exact(contingency_table)
        test_used = "Fisher's Exact Test"
    else:
        # Use Chi-Square test otherwise
        _, p_value, _, _ = chi2_contingency(contingency_table)
        test_used = "Chi-Square test"

    print(f"{test_used} for {column_name}: p-value={p_value}")

# List of columns to test
columns_to_test = [
    'tumor_frontal', 'tumor_temp', 'tumor_parietal',
    'tumor_occipital', 'tumor_insula', 'tumor_subcort'
]

# Perform the tests for each column for pxa_initial_sz vs. pxa_initial_other
print("Comparing pxa_initial_sz vs. pxa_initial_other")
for column in columns_to_test:
    print(f"Testing {column}")
    perform_test(pxa_initial_sz, pxa_initial_other, column)

# Perform the tests for each column for pxa_tre vs. pxa_no_tre
print("\nComparing pxa_tre vs. pxa_no_tre")
for column in columns_to_test:
    print(f"Testing {column}")
    perform_test(pxa_tre, pxa_no_tre, column)

# %% [markdown]
# ### Tumor features: BRAF, CDKN2A/B, TERTp

# %%
def perform_test(group1, group2, column_name):
    # Ensure the column is numeric, converting non-numeric values to NaN and then dropping them
    group1_filtered = group1.copy()
    group1_filtered[column_name] = pd.to_numeric(group1_filtered[column_name], errors='coerce')
    group1_filtered = group1_filtered.dropna(subset=[column_name])

    group2_filtered = group2.copy()
    group2_filtered[column_name] = pd.to_numeric(group2_filtered[column_name], errors='coerce')
    group2_filtered = group2_filtered.dropna(subset=[column_name])

    # Calculate the total numbers of 1s and 0s in each group
    total_1_group1 = (group1_filtered[column_name] == 1).sum()
    total_0_group1 = (group1_filtered[column_name] == 0).sum()

    total_1_group2 = (group2_filtered[column_name] == 1).sum()
    total_0_group2 = (group2_filtered[column_name] == 0).sum()

    # Calculate the proportions
    proportion_1_group1 = total_1_group1 / (total_1_group1 + total_0_group1)
    proportion_1_group2 = total_1_group2 / (total_1_group2 + total_0_group2)

    print(f"Total number of 1s in {column_name} for group1: {total_1_group1}")
    print(f"Total number of 0s in {column_name} for group1: {total_0_group1}")
    print(f"Proportion of 1s in {column_name} for group1: {proportion_1_group1:.2f}")

    print(f"Total number of 1s in {column_name} for group2: {total_1_group2}")
    print(f"Total number of 0s in {column_name} for group2: {total_0_group2}")
    print(f"Proportion of 1s in {column_name} for group2: {proportion_1_group2:.2f}")

    # Create the contingency table
    contingency_table = [
        [total_1_group1, total_0_group1],
        [total_1_group2, total_0_group2]
    ]

    # Perform the Chi-Square test or Fisher's Exact Test
    if min(total_1_group1, total_0_group1, total_1_group2, total_0_group2) < 5:
        # Use Fisher's Exact Test if any expected frequency is less than 5
        _, p_value = fisher_exact(contingency_table)
        test_used = "Fisher's Exact Test"
    else:
        # Use Chi-Square test otherwise
        _, p_value, _, _ = chi2_contingency(contingency_table)
        test_used = "Chi-Square test"

    print(f"{test_used} for {column_name}: p-value={p_value}")

# Perform the tests for each mutation column for pxa_initial_sz vs. pxa_initial_other
print("Comparing pxa_initial_sz vs. pxa_initial_other")
perform_test(pxa_initial_sz, pxa_initial_other, 'mut_BRAF_V600E')
perform_test(pxa_initial_sz, pxa_initial_other, 'mut_CDKN2AB')
perform_test(pxa_initial_sz, pxa_initial_other, 'mut_TERTp')

# Perform the tests for each mutation column for pxa_tre vs. pxa_no_tre
print("\nComparing pxa_tre vs. pxa_no_tre")
perform_test(pxa_tre, pxa_no_tre, 'mut_BRAF_V600E')
perform_test(pxa_tre, pxa_no_tre, 'mut_CDKN2AB')
perform_test(pxa_tre, pxa_no_tre, 'mut_TERTp')

# %% [markdown]
# ## Table 2: Outcomes of those with Epilepsy

# %% [markdown]
# ### Outcome after first resection, compare those with initial seizure only

# %%
# Filter the patients who have a numeric value for sz-free_first_rxn
pxa_initial_sz['sz-free_first_rxn'] = pd.to_numeric(pxa_initial_sz['sz-free_first_rxn'], errors='coerce')
pxa_initial_sz_filtered = pxa_initial_sz.dropna(subset=['sz-free_first_rxn'])

# List of covariates
categorical_covariates = [
    'sex_female', 'tumor_left', 'tumor_frontal', 'tumor_temp', 'tumor_parietal',
    'tumor_occipital', 'tumor_insula', 'tumor_subcort', 'mut_BRAF_V600E',
    'mut_CDKN2AB', 'mut_TERTp', 'tumor_grade_first', 'resection_within_1_year',
    'resection_within_3_years', 'add_chemo', 'add_radiation', 'sz_convulsive', 'sz_drug-resistant',
    'status_epilepticus_preop', 'resection_1_gross_total'
]

continuous_covariates = ['presentation_1_age', 'tumor_size', 'days_after_first_rxn', 'months_after_first_rxn', 'tumor_mutational_burden']

# Ensure all covariates are numeric where necessary
for covariate in categorical_covariates + continuous_covariates:
    pxa_initial_sz_filtered[covariate] = pd.to_numeric(pxa_initial_sz_filtered[covariate], errors='coerce')

# Split the data into seizure-free and not seizure-free groups
seizure_free = pxa_initial_sz_filtered[pxa_initial_sz_filtered['sz-free_first_rxn'] == 1]
not_seizure_free = pxa_initial_sz_filtered[pxa_initial_sz_filtered['sz-free_first_rxn'] == 0]

# Print the total number of patients in each group
print(f"Total number of patients in the seizure-free group: {len(seizure_free)}")
print(f"Total number of patients in the not seizure-free group: {len(not_seizure_free)}\n")

# Function to perform Chi-Square or Fisher's Exact Test for categorical variables
def compare_categorical(group1, group2, column_name):
    if column_name == 'tumor_grade_first':
        total_2_group1 = (group1[column_name] == 2).sum()
        total_3_group1 = (group1[column_name] == 3).sum()
        total_2_group2 = (group2[column_name] == 2).sum()
        total_3_group2 = (group2[column_name] == 3).sum()

        proportion_2_group1 = total_2_group1 / (total_2_group1 + total_3_group1)
        proportion_3_group1 = total_3_group1 / (total_2_group1 + total_3_group1)
        proportion_2_group2 = total_2_group2 / (total_2_group2 + total_3_group2)
        proportion_3_group2 = total_3_group2 / (total_2_group2 + total_3_group2)

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Total 2s = {total_2_group1}, Proportion 2s = {proportion_2_group1:.2f}")
        print(f"Group 1 (Seizure-Free): Total 3s = {total_3_group1}, Proportion 3s = {proportion_3_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Total 2s = {total_2_group2}, Proportion 2s = {proportion_2_group2:.2f}")
        print(f"Group 2 (Not Seizure-Free): Total 3s = {total_3_group2}, Proportion 3s = {proportion_3_group2:.2f}")

        contingency_table = [
            [total_2_group1, total_3_group1],
            [total_2_group2, total_3_group2]
        ]

        if min(total_2_group1, total_3_group1, total_2_group2, total_3_group2) < 5:
            _, p_value = fisher_exact(contingency_table)
            test_used = "Fisher's Exact Test"
        else:
            _, p_value, _, _ = chi2_contingency(contingency_table)
            test_used = "Chi-Square test"

        print(f"{test_used} for {column_name}: p-value={p_value}\n")
    else:
        total_1_group1 = (group1[column_name] == 1).sum()
        total_0_group1 = (group1[column_name] == 0).sum()
        total_1_group2 = (group2[column_name] == 1).sum()
        total_0_group2 = (group2[column_name] == 0).sum()

        proportion_1_group1 = total_1_group1 / (total_1_group1 + total_0_group1)
        proportion_1_group2 = total_1_group2 / (total_1_group2 + total_0_group2)

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Total 1s = {total_1_group1}, Proportion 1s = {proportion_1_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Total 1s = {total_1_group2}, Proportion 1s = {proportion_1_group2:.2f}")

        contingency_table = [
            [total_1_group1, total_0_group1],
            [total_1_group2, total_0_group2]
        ]

        if min(total_1_group1, total_0_group1, total_1_group2, total_0_group2) < 5:
            _, p_value = fisher_exact(contingency_table)
            test_used = "Fisher's Exact Test"
        else:
            _, p_value, _, _ = chi2_contingency(contingency_table)
            test_used = "Chi-Square test"

        print(f"{test_used} for {column_name}: p-value={p_value}\n")

# Function to perform t-test or Mann-Whitney U test for continuous variables
def compare_continuous(group1, group2, column_name):
    group1 = group1.dropna(subset=[column_name])
    group2 = group2.dropna(subset=[column_name])
    
    shapiro_group1 = shapiro(group1[column_name])
    shapiro_group2 = shapiro(group2[column_name])

    if column_name == 'tumor_size':
        mean_group1 = group1[column_name].mean()
        std_group1 = group1[column_name].std()
        mean_group2 = group2[column_name].mean()
        std_group2 = group2[column_name].std()

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Mean = {mean_group1:.2f}, Standard Deviation = {std_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Mean = {mean_group2:.2f}, Standard Deviation = {std_group2:.2f}")

        if shapiro_group1.pvalue > 0.05 and shapiro_group2.pvalue > 0.05:
            t_statistic, p_value = ttest_ind(group1[column_name], group2[column_name])
            test_used = "t-test"
        else:
            u_statistic, p_value = mannwhitneyu(group1[column_name], group2[column_name])
            test_used = "Mann-Whitney U test"

    elif column_name == 'presentation_1_age':
        median_group1 = group1[column_name].median()
        min_group1 = group1[column_name].min()
        max_group1 = group1[column_name].max()
        median_group2 = group2[column_name].median()
        min_group2 = group2[column_name].min()
        max_group2 = group2[column_name].max()

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Median = {median_group1:.2f}, Range = {min_group1:.2f} - {max_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Median = {median_group2:.2f}, Range = {min_group2:.2f} - {max_group2:.2f}")

        if shapiro_group1.pvalue > 0.05 and shapiro_group2.pvalue > 0.05:
            t_statistic, p_value = ttest_ind(group1[column_name], group2[column_name])
            test_used = "t-test"
        else:
            u_statistic, p_value = mannwhitneyu(group1[column_name], group2[column_name])
            test_used = "Mann-Whitney U test"

    elif column_name == 'days_after_first_rxn':
        median_group1 = group1[column_name].median()
        min_group1 = group1[column_name].min()
        max_group1 = group1[column_name].max()
        median_group2 = group2[column_name].median()
        min_group2 = group2[column_name].min()
        max_group2 = group2[column_name].max()

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Median = {median_group1:.2f}, Range = {min_group1:.2f} - {max_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Median = {median_group2:.2f}, Range = {min_group2:.2f} - {max_group2:.2f}")

        if shapiro_group1.pvalue > 0.05 and shapiro_group2.pvalue > 0.05:
            t_statistic, p_value = ttest_ind(group1[column_name], group2[column_name])
            test_used = "t-test"
        else:
            u_statistic, p_value = mannwhitneyu(group1[column_name], group2[column_name])
            test_used = "Mann-Whitney U test"

    elif column_name == 'months_after_first_rxn':
        median_group1 = group1[column_name].median()
        min_group1 = group1[column_name].min()
        max_group1 = group1[column_name].max()
        median_group2 = group2[column_name].median()
        min_group2 = group2[column_name].min()
        max_group2 = group2[column_name].max()

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Median = {median_group1:.2f}, Range = {min_group1:.2f} - {max_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Median = {median_group2:.2f}, Range = {min_group2:.2f} - {max_group2:.2f}")

        if shapiro_group1.pvalue > 0.05 and shapiro_group2.pvalue > 0.05:
            t_statistic, p_value = ttest_ind(group1[column_name], group2[column_name])
            test_used = "t-test"
        else:
            u_statistic, p_value = mannwhitneyu(group1[column_name], group2[column_name])
            test_used = "Mann-Whitney U test"

    elif column_name == 'tumor_mutational_burden':
        mean_group1 = group1[column_name].mean()
        std_group1 = group1[column_name].std()
        mean_group2 = group2[column_name].mean()
        std_group2 = group2[column_name].std()

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Mean = {mean_group1:.2f}, Standard Deviation = {std_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Mean = {mean_group2:.2f}, Standard Deviation = {std_group2:.2f}")

        if shapiro_group1.pvalue > 0.05 and shapiro_group2.pvalue > 0.05:
            t_statistic, p_value = ttest_ind(group1[column_name], group2[column_name])
            test_used = "t-test"
        else:
            u_statistic, p_value = mannwhitneyu(group1[column_name], group2[column_name])
            test_used = "Mann-Whitney U test"

    print(f"{test_used} for {column_name}: p-value={p_value}\n")

# Compare categorical covariates
for covariate in categorical_covariates:
    compare_categorical(seizure_free, not_seizure_free, covariate)

# Compare continuous covariates
for covariate in continuous_covariates:
    compare_continuous(seizure_free, not_seizure_free, covariate)

# %% [markdown]
# ### Outcome after first resection, any seizure

# %%
# Filter the patients with tumor_epilepsy equal to 1
pxa_tre = pxa[pxa['tumor_epilepsy'] == 1]

# Filter the patients who have a numeric value for sz-free_first_rxn
pxa_tre['sz-free_first_rxn'] = pd.to_numeric(pxa_tre['sz-free_first_rxn'], errors='coerce')
pxa_tre_filtered = pxa_tre.dropna(subset=['sz-free_first_rxn'])

# List of covariates
categorical_covariates = [
    'sex_female', 'tumor_left', 'tumor_frontal', 'tumor_temp', 'tumor_parietal',
    'tumor_occipital', 'tumor_insula', 'tumor_subcort', 'mut_BRAF_V600E',
    'mut_CDKN2AB', 'mut_TERTp', 'tumor_grade_first', 'resection_within_1_year',
    'resection_within_3_years', 'add_chemo', 'add_radiation', 'sz_convulsive', 'sz_drug-resistant',
    'status_epilepticus_preop', 'resection_1_gross_total'
]

continuous_covariates = ['presentation_1_age', 'tumor_size', 'days_after_first_rxn', 'months_after_first_rxn', 'tumor_mutational_burden']

# Ensure all covariates are numeric where necessary
for covariate in categorical_covariates + continuous_covariates:
    pxa_tre_filtered[covariate] = pd.to_numeric(pxa_tre_filtered[covariate], errors='coerce')

# Split the data into seizure-free and not seizure-free groups
seizure_free = pxa_tre_filtered[pxa_tre_filtered['sz-free_first_rxn'] == 1]
not_seizure_free = pxa_tre_filtered[pxa_tre_filtered['sz-free_first_rxn'] == 0]

# Print the total number of patients in each group
print(f"Total number of patients in the seizure-free group: {len(seizure_free)}")
print(f"Total number of patients in the not seizure-free group: {len(not_seizure_free)}\n")

# Function to perform Chi-Square or Fisher's Exact Test for categorical variables
def compare_categorical(group1, group2, column_name):
    if column_name == 'tumor_grade_first':
        total_2_group1 = (group1[column_name] == 2).sum()
        total_3_group1 = (group1[column_name] == 3).sum()
        total_2_group2 = (group2[column_name] == 2).sum()
        total_3_group2 = (group2[column_name] == 3).sum()

        proportion_2_group1 = total_2_group1 / (total_2_group1 + total_3_group1)
        proportion_3_group1 = total_3_group1 / (total_2_group1 + total_3_group1)
        proportion_2_group2 = total_2_group2 / (total_2_group2 + total_3_group2)
        proportion_3_group2 = total_3_group2 / (total_2_group2 + total_3_group2)

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Total 2s = {total_2_group1}, Proportion 2s = {proportion_2_group1:.2f}")
        print(f"Group 1 (Seizure-Free): Total 3s = {total_3_group1}, Proportion 3s = {proportion_3_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Total 2s = {total_2_group2}, Proportion 2s = {proportion_2_group2:.2f}")
        print(f"Group 2 (Not Seizure-Free): Total 3s = {total_3_group2}, Proportion 3s = {proportion_3_group2:.2f}")

        contingency_table = [
            [total_2_group1, total_3_group1],
            [total_2_group2, total_3_group2]
        ]

        if min(total_2_group1, total_3_group1, total_2_group2, total_3_group2) < 5:
            _, p_value = fisher_exact(contingency_table)
            test_used = "Fisher's Exact Test"
        else:
            _, p_value, _, _ = chi2_contingency(contingency_table)
            test_used = "Chi-Square test"

        print(f"{test_used} for {column_name}: p-value={p_value}\n")
    else:
        total_1_group1 = (group1[column_name] == 1).sum()
        total_0_group1 = (group1[column_name] == 0).sum()
        total_1_group2 = (group2[column_name] == 1).sum()
        total_0_group2 = (group2[column_name] == 0).sum()

        proportion_1_group1 = total_1_group1 / (total_1_group1 + total_0_group1)
        proportion_1_group2 = total_1_group2 / (total_1_group2 + total_0_group2)

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Total 1s = {total_1_group1}, Proportion 1s = {proportion_1_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Total 1s = {total_1_group2}, Proportion 1s = {proportion_1_group2:.2f}")

        contingency_table = [
            [total_1_group1, total_0_group1],
            [total_1_group2, total_0_group2]
        ]

        if min(total_1_group1, total_0_group1, total_1_group2, total_0_group2) < 5:
            _, p_value = fisher_exact(contingency_table)
            test_used = "Fisher's Exact Test"
        else:
            _, p_value, _, _ = chi2_contingency(contingency_table)
            test_used = "Chi-Square test"

        print(f"{test_used} for {column_name}: p-value={p_value}\n")

# Function to perform t-test or Mann-Whitney U test for continuous variables
def compare_continuous(group1, group2, column_name):
    group1 = group1.dropna(subset=[column_name])
    group2 = group2.dropna(subset=[column_name])
    
    shapiro_group1 = shapiro(group1[column_name])
    shapiro_group2 = shapiro(group2[column_name])

    if column_name == 'tumor_size':
        mean_group1 = group1[column_name].mean()
        std_group1 = group1[column_name].std()
        mean_group2 = group2[column_name].mean()
        std_group2 = group2[column_name].std()

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Mean = {mean_group1:.2f}, Standard Deviation = {std_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Mean = {mean_group2:.2f}, Standard Deviation = {std_group2:.2f}")

        if shapiro_group1.pvalue > 0.05 and shapiro_group2.pvalue > 0.05:
            t_statistic, p_value = ttest_ind(group1[column_name], group2[column_name])
            test_used = "t-test"
        else:
            u_statistic, p_value = mannwhitneyu(group1[column_name], group2[column_name])
            test_used = "Mann-Whitney U test"

    elif column_name == 'presentation_1_age':
        median_group1 = group1[column_name].median()
        min_group1 = group1[column_name].min()
        max_group1 = group1[column_name].max()
        median_group2 = group2[column_name].median()
        min_group2 = group2[column_name].min()
        max_group2 = group2[column_name].max()

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Median = {median_group1:.2f}, Range = {min_group1:.2f} - {max_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Median = {median_group2:.2f}, Range = {min_group2:.2f} - {max_group2:.2f}")

        if shapiro_group1.pvalue > 0.05 and shapiro_group2.pvalue > 0.05:
            t_statistic, p_value = ttest_ind(group1[column_name], group2[column_name])
            test_used = "t-test"
        else:
            u_statistic, p_value = mannwhitneyu(group1[column_name], group2[column_name])
            test_used = "Mann-Whitney U test"

    elif column_name == 'days_after_first_rxn':
        median_group1 = group1[column_name].median()
        min_group1 = group1[column_name].min()
        max_group1 = group1[column_name].max()
        median_group2 = group2[column_name].median()
        min_group2 = group2[column_name].min()
        max_group2 = group2[column_name].max()

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Median = {median_group1:.2f}, Range = {min_group1:.2f} - {max_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Median = {median_group2:.2f}, Range = {min_group2:.2f} - {max_group2:.2f}")

        if shapiro_group1.pvalue > 0.05 and shapiro_group2.pvalue > 0.05:
            t_statistic, p_value = ttest_ind(group1[column_name], group2[column_name])
            test_used = "t-test"
        else:
            u_statistic, p_value = mannwhitneyu(group1[column_name], group2[column_name])
            test_used = "Mann-Whitney U test"

    elif column_name == 'months_after_first_rxn':
        median_group1 = group1[column_name].median()
        min_group1 = group1[column_name].min()
        max_group1 = group1[column_name].max()
        median_group2 = group2[column_name].median()
        min_group2 = group2[column_name].min()
        max_group2 = group2[column_name].max()

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Median = {median_group1:.2f}, Range = {min_group1:.2f} - {max_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Median = {median_group2:.2f}, Range = {min_group2:.2f} - {max_group2:.2f}")

        if shapiro_group1.pvalue > 0.05 and shapiro_group2.pvalue > 0.05:
            t_statistic, p_value = ttest_ind(group1[column_name], group2[column_name])
            test_used = "t-test"
        else:
            u_statistic, p_value = mannwhitneyu(group1[column_name], group2[column_name])
            test_used = "Mann-Whitney U test"

    elif column_name == 'tumor_mutational_burden':
        mean_group1 = group1[column_name].mean()
        std_group1 = group1[column_name].std()
        mean_group2 = group2[column_name].mean()
        std_group2 = group2[column_name].std()

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Mean = {mean_group1:.2f}, Standard Deviation = {std_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Mean = {mean_group2:.2f}, Standard Deviation = {std_group2:.2f}")

        if shapiro_group1.pvalue > 0.05 and shapiro_group2.pvalue > 0.05:
            t_statistic, p_value = ttest_ind(group1[column_name], group2[column_name])
            test_used = "t-test"
        else:
            u_statistic, p_value = mannwhitneyu(group1[column_name], group2[column_name])
            test_used = "Mann-Whitney U test"

    print(f"{test_used} for {column_name}: p-value={p_value}\n")

# Compare categorical covariates
for covariate in categorical_covariates:
    compare_categorical(seizure_free, not_seizure_free, covariate)

# Compare continuous covariates
for covariate in continuous_covariates:
    compare_continuous(seizure_free, not_seizure_free, covariate)

# %% [markdown]
# ### Outcome after last resection

# %%
# Filter the patients who have a numeric value for sz-free_last_rxn
pxa_initial_sz['sz-free_last_rxn'] = pd.to_numeric(pxa_initial_sz['sz-free_last_rxn'], errors='coerce')
pxa_initial_sz_filtered = pxa_initial_sz.dropna(subset=['sz-free_last_rxn'])

# List of covariates
categorical_covariates = [
    'sex_female', 'tumor_left', 'tumor_frontal', 'tumor_temp', 'tumor_parietal',
    'tumor_occipital', 'tumor_insula', 'tumor_subcort', 'mut_BRAF_V600E',
    'mut_CDKN2AB', 'mut_TERTp', 'tumor_grade_last', 'last_rxn_within_1_year',
    'resection_within_3_years', 'any_chemo', 'any_radiation', 'sz_convulsive', 'sz_drug-resistant',
    'status_epilepticus_preop', 'resection_last_gross_total', 'recurrence_any'
]

continuous_covariates = ['presentation_1_age', 'tumor_size', 'days_after_last_rxn', 'months_after_last_rxn', 'recurrence_count', 'tumor_mutational_burden']

# Ensure all covariates are numeric where necessary
for covariate in categorical_covariates + continuous_covariates:
    pxa_initial_sz_filtered[covariate] = pd.to_numeric(pxa_initial_sz_filtered[covariate], errors='coerce')

# Split the data into seizure-free and not seizure-free groups
seizure_free = pxa_initial_sz_filtered[pxa_initial_sz_filtered['sz-free_last_rxn'] == 1]
not_seizure_free = pxa_initial_sz_filtered[pxa_initial_sz_filtered['sz-free_last_rxn'] == 0]

# Print the total number of patients in each group
print(f"Total number of patients in the seizure-free group: {len(seizure_free)}")
print(f"Total number of patients in the not seizure-free group: {len(not_seizure_free)}\n")

# Function to perform Chi-Square or Fisher's Exact Test for categorical variables
def compare_categorical(group1, group2, column_name):
    if column_name == 'tumor_grade_last':
        total_2_group1 = (group1[column_name] == 2).sum()
        total_3_group1 = (group1[column_name] == 3).sum()
        total_2_group2 = (group2[column_name] == 2).sum()
        total_3_group2 = (group2[column_name] == 3).sum()

        proportion_2_group1 = total_2_group1 / (total_2_group1 + total_3_group1)
        proportion_3_group1 = total_3_group1 / (total_2_group1 + total_3_group1)
        proportion_2_group2 = total_2_group2 / (total_2_group2 + total_3_group2)
        proportion_3_group2 = total_3_group2 / (total_2_group2 + total_3_group2)

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Total 2s = {total_2_group1}, Proportion 2s = {proportion_2_group1:.2f}")
        print(f"Group 1 (Seizure-Free): Total 3s = {total_3_group1}, Proportion 3s = {proportion_3_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Total 2s = {total_2_group2}, Proportion 2s = {proportion_2_group2:.2f}")
        print(f"Group 2 (Not Seizure-Free): Total 3s = {total_3_group2}, Proportion 3s = {proportion_3_group2:.2f}")

        contingency_table = [
            [total_2_group1, total_3_group1],
            [total_2_group2, total_3_group2]
        ]

        if min(total_2_group1, total_3_group1, total_2_group2, total_3_group2) < 5:
            _, p_value = fisher_exact(contingency_table)
            test_used = "Fisher's Exact Test"
        else:
            _, p_value, _, _ = chi2_contingency(contingency_table)
            test_used = "Chi-Square test"

        print(f"{test_used} for {column_name}: p-value={p_value}\n")
    else:
        total_1_group1 = (group1[column_name] == 1).sum()
        total_0_group1 = (group1[column_name] == 0).sum()
        total_1_group2 = (group2[column_name] == 1).sum()
        total_0_group2 = (group2[column_name] == 0).sum()

        proportion_1_group1 = total_1_group1 / (total_1_group1 + total_0_group1)
        proportion_1_group2 = total_1_group2 / (total_1_group2 + total_0_group2)

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Total 1s = {total_1_group1}, Proportion 1s = {proportion_1_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Total 1s = {total_1_group2}, Proportion 1s = {proportion_1_group2:.2f}")

        contingency_table = [
            [total_1_group1, total_0_group1],
            [total_1_group2, total_0_group2]
        ]

        if min(total_1_group1, total_0_group1, total_1_group2, total_0_group2) < 5:
            _, p_value = fisher_exact(contingency_table)
            test_used = "Fisher's Exact Test"
        else:
            _, p_value, _, _ = chi2_contingency(contingency_table)
            test_used = "Chi-Square test"

        print(f"{test_used} for {column_name}: p-value={p_value}\n")

# Function to perform t-test or Mann-Whitney U test for continuous variables
def compare_continuous(group1, group2, column_name):
    group1 = group1.dropna(subset=[column_name])
    group2 = group2.dropna(subset=[column_name])
    
    shapiro_group1 = shapiro(group1[column_name])
    shapiro_group2 = shapiro(group2[column_name])

    if column_name == 'tumor_size':
        mean_group1 = group1[column_name].mean()
        std_group1 = group1[column_name].std()
        mean_group2 = group2[column_name].mean()
        std_group2 = group2[column_name].std()

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Mean = {mean_group1:.2f}, Standard Deviation = {std_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Mean = {mean_group2:.2f}, Standard Deviation = {std_group2:.2f}")

        if shapiro_group1.pvalue > 0.05 and shapiro_group2.pvalue > 0.05:
            t_statistic, p_value = ttest_ind(group1[column_name], group2[column_name])
            test_used = "t-test"
        else:
            u_statistic, p_value = mannwhitneyu(group1[column_name], group2[column_name])
            test_used = "Mann-Whitney U test"

    elif column_name == 'presentation_1_age':
        median_group1 = group1[column_name].median()
        min_group1 = group1[column_name].min()
        max_group1 = group1[column_name].max()
        median_group2 = group2[column_name].median()
        min_group2 = group2[column_name].min()
        max_group2 = group2[column_name].max()

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Median = {median_group1:.2f}, Range = {min_group1:.2f} - {max_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Median = {median_group2:.2f}, Range = {min_group2:.2f} - {max_group2:.2f}")

        if shapiro_group1.pvalue > 0.05 and shapiro_group2.pvalue > 0.05:
            t_statistic, p_value = ttest_ind(group1[column_name], group2[column_name])
            test_used = "t-test"
        else:
            u_statistic, p_value = mannwhitneyu(group1[column_name], group2[column_name])
            test_used = "Mann-Whitney U test"

    elif column_name == 'days_after_last_rxn':
        median_group1 = group1[column_name].median()
        min_group1 = group1[column_name].min()
        max_group1 = group1[column_name].max()
        median_group2 = group2[column_name].median()
        min_group2 = group2[column_name].min()
        max_group2 = group2[column_name].max()

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Median = {median_group1:.2f}, Range = {min_group1:.2f} - {max_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Median = {median_group2:.2f}, Range = {min_group2:.2f} - {max_group2:.2f}")

        if shapiro_group1.pvalue > 0.05 and shapiro_group2.pvalue > 0.05:
            t_statistic, p_value = ttest_ind(group1[column_name], group2[column_name])
            test_used = "t-test"
        else:
            u_statistic, p_value = mannwhitneyu(group1[column_name], group2[column_name])
            test_used = "Mann-Whitney U test"

    elif column_name == 'months_after_last_rxn':
        median_group1 = group1[column_name].median()
        min_group1 = group1[column_name].min()
        max_group1 = group1[column_name].max()
        median_group2 = group2[column_name].median()
        min_group2 = group2[column_name].min()
        max_group2 = group2[column_name].max()

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Median = {median_group1:.2f}, Range = {min_group1:.2f} - {max_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Median = {median_group2:.2f}, Range = {min_group2:.2f} - {max_group2:.2f}")

        if shapiro_group1.pvalue > 0.05 and shapiro_group2.pvalue > 0.05:
            t_statistic, p_value = ttest_ind(group1[column_name], group2[column_name])
            test_used = "t-test"
        else:
            u_statistic, p_value = mannwhitneyu(group1[column_name], group2[column_name])
            test_used = "Mann-Whitney U test"

    elif column_name == 'recurrence_count':
        mean_group1 = group1[column_name].mean()
        std_group1 = group1[column_name].std()
        mean_group2 = group2[column_name].mean()
        std_group2 = group2[column_name].std()

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Mean = {mean_group1:.2f}, Standard Deviation = {std_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Mean = {mean_group2:.2f}, Standard Deviation = {std_group2:.2f}")

        if shapiro_group1.pvalue > 0.05 and shapiro_group2.pvalue > 0.05:
            t_statistic, p_value = ttest_ind(group1[column_name], group2[column_name])
            test_used = "t-test"
        else:
            u_statistic, p_value = mannwhitneyu(group1[column_name], group2[column_name])
            test_used = "Mann-Whitney U test"

    elif column_name == 'tumor_mutational_burden':
        mean_group1 = group1[column_name].mean()
        std_group1 = group1[column_name].std()
        mean_group2 = group2[column_name].mean()
        std_group2 = group2[column_name].std()

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Mean = {mean_group1:.2f}, Standard Deviation = {std_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Mean = {mean_group2:.2f}, Standard Deviation = {std_group2:.2f}")

        if shapiro_group1.pvalue > 0.05 and shapiro_group2.pvalue > 0.05:
            t_statistic, p_value = ttest_ind(group1[column_name], group2[column_name])
            test_used = "t-test"
        else:
            u_statistic, p_value = mannwhitneyu(group1[column_name], group2[column_name])
            test_used = "Mann-Whitney U test"

    print(f"{test_used} for {column_name}: p-value={p_value}\n")

# Compare categorical covariates
for covariate in categorical_covariates:
    compare_categorical(seizure_free, not_seizure_free, covariate)

# Compare continuous covariates
for covariate in continuous_covariates:
    compare_continuous(seizure_free, not_seizure_free, covariate)

# %% [markdown]
# ### Outcome after last resection, any seizure  

# %%
# Filter the patients with tumor_epilepsy equal to 1
pxa_tre = pxa[pxa['tumor_epilepsy'] == 1]

# Filter the patients who have a numeric value for sz-free_last_rxn
pxa_tre['sz-free_last_rxn'] = pd.to_numeric(pxa_tre['sz-free_last_rxn'], errors='coerce')
pxa_tre_filtered = pxa_tre.dropna(subset=['sz-free_last_rxn'])

# List of covariates
categorical_covariates = [
    'sex_female', 'tumor_left', 'tumor_frontal', 'tumor_temp', 'tumor_parietal',
    'tumor_occipital', 'tumor_insula', 'tumor_subcort', 'mut_BRAF_V600E',
    'mut_CDKN2AB', 'mut_TERTp', 'tumor_grade_last', 'last_rxn_within_1_year',
    'resection_within_3_years', 'any_chemo', 'any_radiation', 'sz_convulsive', 'sz_drug-resistant',
    'status_epilepticus_preop', 'resection_last_gross_total', 'recurrence_any',
    'tmz_initial', 'braf_initial', 'tmz_any', 'braf_any'
]

continuous_covariates = ['presentation_1_age', 'tumor_size', 'days_after_last_rxn', 'months_after_last_rxn', 'recurrence_count', 'tumor_mutational_burden']

# Ensure all covariates are numeric where necessary
for covariate in categorical_covariates + continuous_covariates:
    pxa_tre_filtered[covariate] = pd.to_numeric(pxa_tre_filtered[covariate], errors='coerce')

# Split the data into seizure-free and not seizure-free groups
seizure_free = pxa_tre_filtered[pxa_tre_filtered['sz-free_last_rxn'] == 1]
not_seizure_free = pxa_tre_filtered[pxa_tre_filtered['sz-free_last_rxn'] == 0]

# Print the total number of patients in each group
print(f"Total number of patients in the seizure-free group: {len(seizure_free)}")
print(f"Total number of patients in the not seizure-free group: {len(not_seizure_free)}\n")

# Function to perform Chi-Square or Fisher's Exact Test for categorical variables
def compare_categorical(group1, group2, column_name):
    group1 = group1.dropna(subset=[column_name])
    group2 = group2.dropna(subset=[column_name])
    
    if column_name == 'tumor_grade_last':
        total_2_group1 = (group1[column_name] == 2).sum()
        total_3_group1 = (group1[column_name] == 3).sum()
        total_2_group2 = (group2[column_name] == 2).sum()
        total_3_group2 = (group2[column_name] == 3).sum()

        proportion_2_group1 = total_2_group1 / (total_2_group1 + total_3_group1)
        proportion_3_group1 = total_3_group1 / (total_2_group1 + total_3_group1)
        proportion_2_group2 = total_2_group2 / (total_2_group2 + total_3_group2)
        proportion_3_group2 = total_3_group2 / (total_2_group2 + total_3_group2)

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Total 2s = {total_2_group1}, Proportion 2s = {proportion_2_group1:.2f}")
        print(f"Group 1 (Seizure-Free): Total 3s = {total_3_group1}, Proportion 3s = {proportion_3_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Total 2s = {total_2_group2}, Proportion 2s = {proportion_2_group2:.2f}")
        print(f"Group 2 (Not Seizure-Free): Total 3s = {total_3_group2}, Proportion 3s = {proportion_3_group2:.2f}")

        contingency_table = [
            [total_2_group1, total_3_group1],
            [total_2_group2, total_3_group2]
        ]

        if min(total_2_group1, total_3_group1, total_2_group2, total_3_group2) < 5:
            _, p_value = fisher_exact(contingency_table)
            test_used = "Fisher's Exact Test"
        else:
            _, p_value, _, _ = chi2_contingency(contingency_table)
            test_used = "Chi-Square test"

        print(f"{test_used} for {column_name}: p-value={p_value}\n")
    else:
        total_1_group1 = (group1[column_name] == 1).sum()
        total_0_group1 = (group1[column_name] == 0).sum()
        total_1_group2 = (group2[column_name] == 1).sum()
        total_0_group2 = (group2[column_name] == 0).sum()

        proportion_1_group1 = total_1_group1 / (total_1_group1 + total_0_group1)
        proportion_1_group2 = total_1_group2 / (total_1_group2 + total_0_group2)

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Total 1s = {total_1_group1}, Proportion 1s = {proportion_1_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Total 1s = {total_1_group2}, Proportion 1s = {proportion_1_group2:.2f}")

        contingency_table = [
            [total_1_group1, total_0_group1],
            [total_1_group2, total_0_group2]
        ]

        if min(total_1_group1, total_0_group1, total_1_group2, total_0_group2) < 5:
            _, p_value = fisher_exact(contingency_table)
            test_used = "Fisher's Exact Test"
        else:
            _, p_value, _, _ = chi2_contingency(contingency_table)
            test_used = "Chi-Square test"

        print(f"{test_used} for {column_name}: p-value={p_value}\n")

# Function to perform t-test or Mann-Whitney U test for continuous variables
def compare_continuous(group1, group2, column_name):
    group1 = group1.dropna(subset=[column_name])
    group2 = group2.dropna(subset=[column_name])
    
    shapiro_group1 = shapiro(group1[column_name])
    shapiro_group2 = shapiro(group2[column_name])

    if column_name == 'tumor_size':
        mean_group1 = group1[column_name].mean()
        std_group1 = group1[column_name].std()
        mean_group2 = group2[column_name].mean()
        std_group2 = group2[column_name].std()

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Mean = {mean_group1:.2f}, Standard Deviation = {std_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Mean = {mean_group2:.2f}, Standard Deviation = {std_group2:.2f}")

        if shapiro_group1.pvalue > 0.05 and shapiro_group2.pvalue > 0.05:
            t_statistic, p_value = ttest_ind(group1[column_name], group2[column_name])
            test_used = "t-test"
        else:
            u_statistic, p_value = mannwhitneyu(group1[column_name], group2[column_name])
            test_used = "Mann-Whitney U test"

    elif column_name == 'presentation_1_age':
        median_group1 = group1[column_name].median()
        min_group1 = group1[column_name].min()
        max_group1 = group1[column_name].max()
        median_group2 = group2[column_name].median()
        min_group2 = group2[column_name].min()
        max_group2 = group2[column_name].max()

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Median = {median_group1:.2f}, Range = {min_group1:.2f} - {max_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Median = {median_group2:.2f}, Range = {min_group2:.2f} - {max_group2:.2f}")

        if shapiro_group1.pvalue > 0.05 and shapiro_group2.pvalue > 0.05:
            t_statistic, p_value = ttest_ind(group1[column_name], group2[column_name])
            test_used = "t-test"
        else:
            u_statistic, p_value = mannwhitneyu(group1[column_name], group2[column_name])
            test_used = "Mann-Whitney U test"

    elif column_name == 'days_after_last_rxn':
        median_group1 = group1[column_name].median()
        min_group1 = group1[column_name].min()
        max_group1 = group1[column_name].max()
        median_group2 = group2[column_name].median()
        min_group2 = group2[column_name].min()
        max_group2 = group2[column_name].max()

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Median = {median_group1:.2f}, Range = {min_group1:.2f} - {max_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Median = {median_group2:.2f}, Range = {min_group2:.2f} - {max_group2:.2f}")

        if shapiro_group1.pvalue > 0.05 and shapiro_group2.pvalue > 0.05:
            t_statistic, p_value = ttest_ind(group1[column_name], group2[column_name])
            test_used = "t-test"
        else:
            u_statistic, p_value = mannwhitneyu(group1[column_name], group2[column_name])
            test_used = "Mann-Whitney U test"

    elif column_name == 'months_after_last_rxn':
        median_group1 = group1[column_name].median()
        min_group1 = group1[column_name].min()
        max_group1 = group1[column_name].max()
        median_group2 = group2[column_name].median()
        min_group2 = group2[column_name].min()
        max_group2 = group2[column_name].max()

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Median = {median_group1:.2f}, Range = {min_group1:.2f} - {max_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Median = {median_group2:.2f}, Range = {min_group2:.2f} - {max_group2:.2f}")

        if shapiro_group1.pvalue > 0.05 and shapiro_group2.pvalue > 0.05:
            t_statistic, p_value = ttest_ind(group1[column_name], group2[column_name])
            test_used = "t-test"
        else:
            u_statistic, p_value = mannwhitneyu(group1[column_name], group2[column_name])
            test_used = "Mann-Whitney U test"

    elif column_name == 'recurrence_count':
        mean_group1 = group1[column_name].mean()
        std_group1 = group1[column_name].std()
        mean_group2 = group2[column_name].mean()
        std_group2 = group2[column_name].std()

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Mean = {mean_group1:.2f}, Standard Deviation = {std_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Mean = {mean_group2:.2f}, Standard Deviation = {std_group2:.2f}")

        if shapiro_group1.pvalue > 0.05 and shapiro_group2.pvalue > 0.05:
            t_statistic, p_value = ttest_ind(group1[column_name], group2[column_name])
            test_used = "t-test"
        else:
            u_statistic, p_value = mannwhitneyu(group1[column_name], group2[column_name])
            test_used = "Mann-Whitney U test"

    elif column_name == 'tumor_mutational_burden':
        mean_group1 = group1[column_name].mean()
        std_group1 = group1[column_name].std()
        mean_group2 = group2[column_name].mean()
        std_group2 = group2[column_name].std()

        print(f"Covariate: {column_name}")
        print(f"Group 1 (Seizure-Free): Mean = {mean_group1:.2f}, Standard Deviation = {std_group1:.2f}")
        print(f"Group 2 (Not Seizure-Free): Mean = {mean_group2:.2f}, Standard Deviation = {std_group2:.2f}")

        if shapiro_group1.pvalue > 0.05 and shapiro_group2.pvalue > 0.05:
            t_statistic, p_value = ttest_ind(group1[column_name], group2[column_name])
            test_used = "t-test"
        else:
            u_statistic, p_value = mannwhitneyu(group1[column_name], group2[column_name])
            test_used = "Mann-Whitney U test"

    print(f"{test_used} for {column_name}: p-value={p_value}\n")

# Compare categorical covariates
for covariate in categorical_covariates:
    compare_categorical(seizure_free, not_seizure_free, covariate)

# Compare continuous covariates
for covariate in continuous_covariates:
    compare_continuous(seizure_free, not_seizure_free, covariate)

# %% [markdown]
# ## Figure 1. Survival Analysis

# %% [markdown]
# ### Survival analysis for with TRE vs. without TRE

# %%
# Create Kaplan-Meier fitter instances
kmf_tre = KaplanMeierFitter()
kmf_no_tre = KaplanMeierFitter()

# Fit the data for pxa_tre
kmf_tre.fit(durations=pxa_tre['survival_months'], event_observed=pxa_tre['death'], label='PXA with TRE')

# Fit the data for pxa_no_tre
kmf_no_tre.fit(durations=pxa_no_tre['survival_months'], event_observed=pxa_no_tre['death'], label='PXA without TRE')

# Plot the survival curves with censored patients marked and a survival table
plt.figure(figsize=(12, 8))
ax = kmf_tre.plot_survival_function(show_censors=True, ci_show=True, color='blue', ax=None)
kmf_no_tre.plot_survival_function(show_censors=True, ci_show=True, color='orange', ax=ax)
plt.xlabel('Months')
plt.ylabel('Survival Probability')
plt.title('Kaplan-Meier Survival Curves')
plt.xlim(0, 407)  # Expand the x-axis to 407 months
plt.xticks(range(0, 408, 50))  # Set x-ticks to cover up to 407 months with intervals of 50 months
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add the survival table
from lifelines.plotting import add_at_risk_counts
add_at_risk_counts(kmf_tre, kmf_no_tre, ax=ax)
plt.tight_layout()
plt.show()

# Perform the log-rank test
results = logrank_test(pxa_tre['survival_months'], pxa_no_tre['survival_months'], event_observed_A=pxa_tre['death'], event_observed_B=pxa_no_tre['death'])
print(f"Log-rank test p-value: {results.p_value}")

# %% [markdown]
# ### Add multiple comparisons

# %%
# Assuming pxa_tre and pxa_no_tre are your DataFrames

# Add a 'group' column to distinguish between the two groups
pxa_tre['group'] = 1  # Seizure-Free group
pxa_no_tre['group'] = 0  # Not Seizure-Free group

# Combine the two DataFrames
pxa = pd.concat([pxa_tre, pxa_no_tre], ignore_index=True)

# Ensure continuous variables are numeric, converting non-numeric values to NaN
continuous_vars = ['presentation_1_age', 'tumor_size']
for var in continuous_vars:
    pxa[var] = pd.to_numeric(pxa[var], errors='coerce')

# Ensure categorical variables are properly formatted
categorical_vars = ['tumor_left', 'sex_female', 'tumor_frontal', 'tumor_temp', 
                    'tumor_parietal', 'tumor_occipital', 'tumor_insula', 'tumor_subcort', 'mut_BRAF_V600E', 
                    'tumor_grade_first_2']

# Convert categorical variables to numeric values
for var in categorical_vars:
    pxa[var] = pd.to_numeric(pxa[var], errors='coerce')

# Function to perform statistical tests for continuous variables
def analyze_continuous_var(var):
    data = pxa.dropna(subset=[var]).copy()
    group1 = data[data['group'] == 1][var]
    group2 = data[data['group'] == 0][var]
    
    # Perform t-test or Mann-Whitney U test
    if np.all(np.isfinite(group1)) and np.all(np.isfinite(group2)):
        t_stat, p_value = ttest_ind(group1, group2)
    else:
        t_stat, p_value = mannwhitneyu(group1, group2)
    
    print(f"Covariate: {var}")
    print(f"Group 1 (Seizure-Free): Mean = {group1.mean()}, Standard Deviation = {group1.std()}")
    print(f"Group 2 (Not Seizure-Free): Mean = {group2.mean()}, Standard Deviation = {group2.std()}")
    print(f"t-test/Mann-Whitney U test for {var}: p-value={p_value}\n")

# Function to perform statistical tests for categorical variables
def analyze_categorical_var(var):
    data = pxa.dropna(subset=[var]).copy()
    
    # Drop rows with 'NR' in the categorical variables
    if var in ['mut_BRAF_V600E', 'tumor_grade_first_2']:
        data = data[data[var] != 'NR']
    
    group1 = data[data['group'] == 1][var]
    group2 = data[data['group'] == 0][var]
    
    # Debugging statements to check the data
    print(f"Analyzing {var}")
    print(f"Group 1 (Seizure-Free): {group1.value_counts()}")
    print(f"Group 2 (Not Seizure-Free): {group2.value_counts()}")
    
    # Calculate the total numbers of 1s and 0s in each group
    total_1_group1 = group1.sum()
    total_0_group1 = len(group1) - total_1_group1
    total_1_group2 = group2.sum()
    total_0_group2 = len(group2) - total_1_group2
    
    # Create the contingency table
    contingency_table = [
        [total_1_group1, total_0_group1],
        [total_1_group2, total_0_group2]
    ]
    
    # Perform Fisher's Exact Test
    _, p_value = fisher_exact(contingency_table)
    
    print(f"Covariate: {var}")
    print(f"Group 1 (Seizure-Free): Total 1s = {total_1_group1}, Proportion 1s = {total_1_group1 / (total_1_group1 + total_0_group1):.2f}")
    print(f"Group 2 (Not Seizure-Free): Total 1s = {total_1_group2}, Proportion 1s = {total_1_group2 / (total_1_group2 + total_0_group2):.2f}")
    print(f"Fisher's Exact Test for {var}: p-value={p_value}\n")

# Analyze continuous variables
for var in continuous_vars:
    analyze_continuous_var(var)

# Analyze categorical variables
for var in categorical_vars:
    if var in pxa.columns:
        analyze_categorical_var(var)
    else:
        print(f"Variable {var} is not present in the dataset and will be skipped.\n")

# Perform multiple comparisons analysis of the log-rank test between the pxa_tre and pxa_no_tre groups
results = logrank_test(pxa_tre['survival_months'], pxa_no_tre['survival_months'], event_observed_A=pxa_tre['death'], event_observed_B=pxa_no_tre['death'])
print(f"Log-rank test p-value: {results.p_value}")

# Fit a Cox proportional hazards model to adjust for covariates
coxph = CoxPHFitter()
# Select relevant columns for the Cox model
cox_columns = ['survival_months', 'death', 'group'] + continuous_vars + categorical_vars
# Drop rows with any NaN values in the selected columns
cox_data = pxa[cox_columns].dropna()

# Fit the Cox model
coxph.fit(cox_data, duration_col='survival_months', event_col='death')

# Print the summary of the Cox model
coxph.print_summary()

# Extract the p-value for the 'group' variable (treatment effect)
adjusted_p_value = coxph.summary.loc['group', 'p']
print(f"Adjusted p-value for survival between pxa_tre and pxa_no_tre: {adjusted_p_value}")

# %%
import pandas as pd

# Assuming the Cox model has already been fitted as `coxph`

# Extract the relevant columns from the Cox model summary
cox_summary = coxph.summary
cox_summary['Hazard Ratio'] = cox_summary['exp(coef)']
cox_summary['95% CI'] = cox_summary.apply(
    lambda row: f"{row['exp(coef) lower 95%']:.2f} - {row['exp(coef) upper 95%']:.2f}", axis=1
)
cox_summary['p-value'] = cox_summary['p']

# Create the modified table
modified_table = cox_summary[['Hazard Ratio', '95% CI', 'p-value']].reset_index()
modified_table.rename(columns={'index': 'Variable'}, inplace=True)

# Print the modified table
print(modified_table)

# %% [markdown]
# ### Tumor-directed therapy and survival

# %%
from lifelines import CoxPHFitter

# Function to calculate hazard ratio and p-value using Cox proportional hazards model
def cox_analysis(data, variable):
    # Prepare the data for the Cox model
    cox_data = data[['survival_months', 'death', variable]].dropna()
    
    # Fit the Cox model
    coxph = CoxPHFitter()
    coxph.fit(cox_data, duration_col='survival_months', event_col='death')
    
    # Extract the hazard ratio and p-value
    hazard_ratio = coxph.summary.loc[variable, 'exp(coef)']
    p_value = coxph.summary.loc[variable, 'p']
    
    # Print the results
    print(f"Variable: {variable}")
    print(f"Hazard Ratio: {hazard_ratio:.2f}")
    print(f"p-value: {p_value:.4f}\n")

# Analyze survival for tmz_initial, tmz_any, braf_initial, and braf_any
variables_to_analyze = ['tmz_initial', 'tmz_any', 'braf_initial', 'braf_any']
for var in variables_to_analyze:
    cox_analysis(pxa, var)

# %%
# Function to calculate mean, standard deviation, and p-value for survival months
def analyze_survival_by_tre(data_tre, data_no_tre, variable):
    # Filter data for the variable
    tre_with_var = data_tre[data_tre[variable] == 1]
    tre_without_var = data_tre[data_tre[variable] == 0]
    no_tre_with_var = data_no_tre[data_no_tre[variable] == 1]
    no_tre_without_var = data_no_tre[data_no_tre[variable] == 0]
    
    # Calculate mean and standard deviation for survival months
    mean_tre_with_var = tre_with_var['survival_months'].mean()
    std_tre_with_var = tre_with_var['survival_months'].std()
    mean_tre_without_var = tre_without_var['survival_months'].mean()
    std_tre_without_var = tre_without_var['survival_months'].std()
    mean_no_tre_with_var = no_tre_with_var['survival_months'].mean()
    std_no_tre_with_var = no_tre_with_var['survival_months'].std()
    mean_no_tre_without_var = no_tre_without_var['survival_months'].mean()
    std_no_tre_without_var = no_tre_without_var['survival_months'].std()
    
    # Perform t-tests
    t_stat_tre, p_value_tre = ttest_ind(tre_with_var['survival_months'], tre_without_var['survival_months'], nan_policy='omit')
    t_stat_no_tre, p_value_no_tre = ttest_ind(no_tre_with_var['survival_months'], no_tre_without_var['survival_months'], nan_policy='omit')
    
    # Print results
    print(f"Variable: {variable}")
    print(f"With TRE and {variable}: Mean = {mean_tre_with_var:.2f} months, SD = {std_tre_with_var:.2f} months")
    print(f"With TRE and no {variable}: Mean = {mean_tre_without_var:.2f} months, SD = {std_tre_without_var:.2f} months")
    print(f"t-test p-value (TRE): {p_value_tre:.4f}")
    print(f"Without TRE and {variable}: Mean = {mean_no_tre_with_var:.2f} months, SD = {std_no_tre_with_var:.2f} months")
    print(f"Without TRE and no {variable}: Mean = {mean_no_tre_without_var:.2f} months, SD = {std_no_tre_without_var:.2f} months")
    print(f"t-test p-value (No TRE): {p_value_no_tre:.4f}\n")

# Analyze survival for tmz_any and braf_any
variables_to_analyze = ['tmz_any', 'braf_any']
for var in variables_to_analyze:
    analyze_survival_by_tre(pxa_tre, pxa_no_tre, var)


