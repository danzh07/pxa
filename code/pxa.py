# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind, shapiro, chi2_contingency, fisher_exact
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from loguru import logger
from pathlib import Path

# Configure loguru
logger.remove()  # Remove default handler

# This function will be updated to accept an output directory
def main(output_dir=None):
    """
    Main execution function for PXA analysis.
    
    Parameters:
    -----------
    output_dir : str
        Directory where logs and figures will be saved
    """
    # Set default output directory if none provided
    if output_dir is None:
        output_dir = '/Users/nishant/Dropbox/Sinha/Lab/Publications/2025/DanPXA/pxa/figures'
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure logger to save to the output directory
    log_path = os.path.join(output_dir, "pxa_analysis.log")
    logger.add(log_path, rotation="10 MB", level="INFO", format="{time} | {level} | {message}")
    logger.add(lambda msg: print(msg), level="INFO", format="{message}")  # Console output without timestamp
    
    logger.info(f"Output directory set to: {output_dir}")

    # Load data
    pxa = load_data()
    
    # Create groups
    pxa_initial_sz, pxa_initial_other, pxa_tre, pxa_no_tre = create_groups(pxa)
    
    # Run analyses
    analyze_clinical_characteristics(pxa_initial_sz, pxa_initial_other, pxa_tre, pxa_no_tre)
    analyze_seizure_outcomes(pxa_initial_sz, pxa_tre)
    
    # Add survival analysis with figure saving
    perform_survival_analysis(pxa_tre, pxa_no_tre, output_dir)
    
    logger.info("Analysis complete!")

# Data loading and preprocessing functions
def load_data(project_path=None):
    """Load and preprocess the PXA dataset."""
    if project_path is None:
        project_path = os.path.dirname(os.getcwd())
    
    pxa_path = os.path.join(project_path, 'data', 'pxa_data.csv')
    logger.info(f"Loading data from {pxa_path}")
    
    pxa = pd.read_csv(pxa_path)
    pxa.set_index('ID', inplace=True)
    
    logger.info(f"Loaded data with {len(pxa)} patients")
    return pxa

def create_groups(pxa):
    """Create patient groups based on seizure status."""
    pxa_initial_sz = pxa[pxa['first_tumor_sz'] == 1]
    pxa_initial_other = pxa[pxa['first_tumor_sz'] == 0]
    pxa_tre = pxa[pxa['tumor_epilepsy'] == 1]
    pxa_no_tre = pxa[pxa['tumor_epilepsy'] == 0]
    
    logger.info(f"Created groups: initial seizure ({len(pxa_initial_sz)} patients), initial other ({len(pxa_initial_other)} patients)")
    logger.info(f"Created groups: tumor epilepsy ({len(pxa_tre)} patients), no tumor epilepsy ({len(pxa_no_tre)} patients)")
    
    return pxa_initial_sz, pxa_initial_other, pxa_tre, pxa_no_tre

# Statistical test functions
def filter_numeric(df, column):
    """Filter dataframe for numeric values in the specified column."""
    df_filtered = df.copy()
    df_filtered[column] = pd.to_numeric(df_filtered[column], errors='coerce')
    return df_filtered.dropna(subset=[column])

def check_normality(df, column):
    """Check normality of data using Shapiro-Wilk test."""
    return shapiro(df[column])

def calculate_median_and_range(df, column):
    """Calculate median and range for a column."""
    median = df[column].median()
    min_val = df[column].min()
    max_val = df[column].max()
    return median, min_val, max_val

def calculate_mean_and_std(df, column):
    """Calculate mean and standard deviation for a column."""
    mean = df[column].mean()
    std = df[column].std()
    return mean, std

def perform_statistical_test(df1, df2, column):
    """Perform appropriate statistical test based on normality."""
    shapiro1 = check_normality(df1, column)
    shapiro2 = check_normality(df2, column)
    
    if shapiro1.pvalue > 0.05 and shapiro2.pvalue > 0.05:
        t_statistic, p_value = ttest_ind(df1[column], df2[column])
        test_used = "t-test"
    else:
        u_statistic, p_value = mannwhitneyu(df1[column], df2[column])
        test_used = "Mann-Whitney U test"
        
    return test_used, t_statistic if test_used == "t-test" else u_statistic, p_value

def perform_categorical_test(group1, group2, column_name):
    """Perform categorical statistical test (Chi-Square or Fisher's Exact)."""
    # Ensure the column is numeric
    group1_filtered = filter_numeric(group1, column_name)
    group2_filtered = filter_numeric(group2, column_name)

    # Calculate counts
    total_1_group1 = (group1_filtered[column_name] == 1).sum()
    total_0_group1 = (group1_filtered[column_name] == 0).sum()
    total_1_group2 = (group2_filtered[column_name] == 1).sum()
    total_0_group2 = (group2_filtered[column_name] == 0).sum()

    # Calculate proportions
    proportion_1_group1 = total_1_group1 / (total_1_group1 + total_0_group1)
    proportion_1_group2 = total_1_group2 / (total_1_group2 + total_0_group2)

    logger.info(f"{column_name}: Group 1 - {total_1_group1}/{total_1_group1 + total_0_group1} ({proportion_1_group1:.2f})")
    logger.info(f"{column_name}: Group 2 - {total_1_group2}/{total_1_group2 + total_0_group2} ({proportion_1_group2:.2f})")

    # Create contingency table
    contingency_table = [
        [total_1_group1, total_0_group1],
        [total_1_group2, total_0_group2]
    ]

    # Choose appropriate test
    if min(total_1_group1, total_0_group1, total_1_group2, total_0_group2) < 5:
        _, p_value = fisher_exact(contingency_table)
        test_used = "Fisher's Exact Test"
    else:
        _, p_value, _, _ = chi2_contingency(contingency_table)
        test_used = "Chi-Square test"

    logger.info(f"{test_used} for {column_name}: p-value={p_value:.4f}")
    return test_used, p_value, proportion_1_group1, proportion_1_group2

def compare_continuous_variables(group1, group2, column, description=""):
    """Compare continuous variables between two groups."""
    group1_filtered = filter_numeric(group1, column)
    group2_filtered = filter_numeric(group2, column)
    
    if column == 'presentation_1_age':
        # Age analysis
        median1, min1, max1 = calculate_median_and_range(group1_filtered, column)
        median2, min2, max2 = calculate_median_and_range(group2_filtered, column)
        
        logger.info(f"{description} Age: Group 1 - Median: {median1:.1f} (Range: {min1:.1f}-{max1:.1f})")
        logger.info(f"{description} Age: Group 2 - Median: {median2:.1f} (Range: {min2:.1f}-{max2:.1f})")
    
    elif column == 'tumor_size' or column == 'tumor_mutational_burden':
        # Size or TMB analysis
        mean1, std1 = calculate_mean_and_std(group1_filtered, column)
        mean2, std2 = calculate_mean_and_std(group2_filtered, column)
        
        logger.info(f"{description} {column}: Group 1 - Mean: {mean1:.2f} (SD: {std1:.2f})")
        logger.info(f"{description} {column}: Group 2 - Mean: {mean2:.2f} (SD: {std2:.2f})")
    
    # Check normality
    shapiro1 = check_normality(group1_filtered, column)
    shapiro2 = check_normality(group2_filtered, column)
    
    # Perform appropriate test
    if shapiro1.pvalue > 0.05 and shapiro2.pvalue > 0.05:
        t_statistic, p_value = ttest_ind(group1_filtered[column], group2_filtered[column])
        test_used = "t-test"
    else:
        u_statistic, p_value = mannwhitneyu(group1_filtered[column], group2_filtered[column])
        test_used = "Mann-Whitney U test"
    
    logger.info(f"{test_used} for {column}: p-value={p_value:.4f}")
    return test_used, p_value

# Analysis functions for different tables
def analyze_clinical_characteristics(pxa_initial_sz, pxa_initial_other, pxa_tre, pxa_no_tre):
    """Analyze clinical characteristics for Table 1."""
    logger.info("=== TABLE 1: CLINICAL CHARACTERISTICS ===")
    
    # Group totals
    total_patients = len(pxa_initial_sz) + len(pxa_initial_other)
    
    logger.info(f"Total patients: {total_patients}")
    logger.info(f"Initial seizure: {len(pxa_initial_sz)} ({len(pxa_initial_sz)/total_patients:.2f})")
    logger.info(f"Initial other symptoms: {len(pxa_initial_other)} ({len(pxa_initial_other)/total_patients:.2f})")
    logger.info(f"Tumor epilepsy: {len(pxa_tre)} ({len(pxa_tre)/total_patients:.2f})")
    logger.info(f"No tumor epilepsy: {len(pxa_no_tre)} ({len(pxa_no_tre)/total_patients:.2f})")
    
    # Analyze continuous variables
    logger.info("\n=== CONTINUOUS VARIABLES ===")
    
    # Age analysis
    compare_continuous_variables(pxa_initial_sz, pxa_initial_other, 'presentation_1_age', "Initial symptom:")
    compare_continuous_variables(pxa_tre, pxa_no_tre, 'presentation_1_age', "Tumor epilepsy:")
    
    # Tumor size analysis
    compare_continuous_variables(pxa_initial_sz, pxa_initial_other, 'tumor_size', "Initial symptom:")
    compare_continuous_variables(pxa_tre, pxa_no_tre, 'tumor_size', "Tumor epilepsy:")
    
    # Tumor mutation burden analysis
    compare_continuous_variables(pxa_initial_sz, pxa_initial_other, 'tumor_mutational_burden', "Initial symptom:")
    compare_continuous_variables(pxa_tre, pxa_no_tre, 'tumor_mutational_burden', "Tumor epilepsy:")
    
    # Categorical variables
    logger.info("\n=== CATEGORICAL VARIABLES ===")
    
    # Sex analysis
    logger.info("Comparing sex distribution:")
    perform_categorical_test(pxa_initial_sz, pxa_initial_other, 'sex_female')
    perform_categorical_test(pxa_tre, pxa_no_tre, 'sex_female')
    
    # Tumor locations
    location_columns = ['tumor_frontal', 'tumor_temp', 'tumor_parietal',
                       'tumor_occipital', 'tumor_insula', 'tumor_subcort']
    
    logger.info("\n=== TUMOR LOCATIONS ===")
    for column in location_columns:
        logger.info(f"Analyzing {column}:")
        perform_categorical_test(pxa_initial_sz, pxa_initial_other, column)
        perform_categorical_test(pxa_tre, pxa_no_tre, column)
    
    # Tumor features
    feature_columns = ['mut_BRAF_V600E', 'mut_CDKN2AB', 'mut_TERTp']
    
    logger.info("\n=== TUMOR FEATURES ===")
    for column in feature_columns:
        logger.info(f"Analyzing {column}:")
        perform_categorical_test(pxa_initial_sz, pxa_initial_other, column)
        perform_categorical_test(pxa_tre, pxa_no_tre, column)

def analyze_seizure_outcomes(pxa_initial_sz, pxa_tre):
    """Analyze seizure outcome (Table 2)."""
    logger.info("\n=== TABLE 2: SEIZURE OUTCOMES ===")
    
    # Outcome after first resection for initial seizure patients
    logger.info("\n=== OUTCOME AFTER FIRST RESECTION (INITIAL SEIZURE) ===")
    pxa_initial_sz = filter_numeric(pxa_initial_sz, 'sz-free_first_rxn')
    seizure_free = pxa_initial_sz[pxa_initial_sz['sz-free_first_rxn'] == 1]
    not_seizure_free = pxa_initial_sz[pxa_initial_sz['sz-free_first_rxn'] == 0]
    
    logger.info(f"Seizure free after first resection: {len(seizure_free)}/{len(pxa_initial_sz)} ({len(seizure_free)/len(pxa_initial_sz):.2f})")
    logger.info(f"Not seizure free after first resection: {len(not_seizure_free)}/{len(pxa_initial_sz)} ({len(not_seizure_free)/len(pxa_initial_sz):.2f})")
    
    # Factors associated with seizure outcome
    categorical_factors = [
        'sex_female', 'tumor_left', 'tumor_frontal', 'tumor_temp', 
        'tumor_parietal', 'tumor_occipital', 'tumor_insula', 'tumor_subcort', 
        'mut_BRAF_V600E', 'mut_CDKN2AB', 'mut_TERTp', 'tumor_grade_first',
        'resection_1_gross_total', 'add_chemo', 'add_radiation'
    ]
    
    continuous_factors = [
        'presentation_1_age', 'tumor_size', 'tumor_mutational_burden'
    ]
    
    # Analyze factors for initial seizure patients
    logger.info("\nFactors associated with seizure freedom after first resection (initial seizure patients):")
    
    for factor in categorical_factors:
        logger.info(f"Analyzing {factor}:")
        perform_categorical_test(seizure_free, not_seizure_free, factor)
    
    for factor in continuous_factors:
        logger.info(f"Analyzing {factor}:")
        compare_continuous_variables(seizure_free, not_seizure_free, factor)
        
    # Repeat for tumor epilepsy patients
    logger.info("\n=== OUTCOME AFTER FIRST RESECTION (ANY SEIZURE) ===")
    pxa_tre = filter_numeric(pxa_tre, 'sz-free_first_rxn')
    seizure_free = pxa_tre[pxa_tre['sz-free_first_rxn'] == 1]
    not_seizure_free = pxa_tre[pxa_tre['sz-free_first_rxn'] == 0]
    
    logger.info(f"Seizure free after first resection: {len(seizure_free)}/{len(pxa_tre)} ({len(seizure_free)/len(pxa_tre):.2f})")
    logger.info(f"Not seizure free after first resection: {len(not_seizure_free)}/{len(pxa_tre)} ({len(not_seizure_free)/len(pxa_tre):.2f})")
    
    # Analyze factors for all seizure patients
    logger.info("\nFactors associated with seizure freedom after first resection (all seizure patients):")
    
    for factor in categorical_factors:
        logger.info(f"Analyzing {factor}:")
        perform_categorical_test(seizure_free, not_seizure_free, factor)
    
    for factor in continuous_factors:
        logger.info(f"Analyzing {factor}:")
        compare_continuous_variables(seizure_free, not_seizure_free, factor)

# Updating the save_figure function to use the passed output directory
def save_figure(fig, filename, folder=None, dpi=300, formats=None):
    """
    Save a matplotlib figure to the specified folder with given filename.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to save
    filename : str
        The base filename (without extension)
    folder : str
        The folder to save figures in (will be created if it doesn't exist)
    dpi : int
        Resolution for the saved figure
    formats : list
        List of formats to save (e.g., ['png', 'svg'])
        If None, defaults to ['png']
    """
    # Default formats if none specified
    if formats is None:
        formats = ['png']  # Only save PNG by default
    
    # Create the figures directory if it doesn't exist
    folder_path = Path(folder)
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
        logger.info(f"Created directory: {folder}")
    
    # Save the figure in each format
    for fmt in formats:
        save_path = folder_path / f"{filename}.{fmt}"
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")

# Updating the survival analysis function to accept output_dir parameter
def perform_survival_analysis(pxa_tre, pxa_no_tre, output_dir=None):
    """
    Perform survival analysis comparing TRE vs no TRE groups.
    
    Parameters:
    -----------
    pxa_tre : pandas.DataFrame
        DataFrame containing patients with tumor-related epilepsy
    pxa_no_tre : pandas.DataFrame
        DataFrame containing patients without tumor-related epilepsy
    output_dir : str
        Directory where figures will be saved
    """
    logger.info("\n=== SURVIVAL ANALYSIS ===")
    
    # Create Kaplan-Meier fitter instances
    kmf_tre = KaplanMeierFitter()
    kmf_no_tre = KaplanMeierFitter()
    
    # Fit the data for pxa_tre
    kmf_tre.fit(durations=pxa_tre['survival_months'], 
                event_observed=pxa_tre['death'], 
                label='PXA with TRE')
    
    # Fit the data for pxa_no_tre
    kmf_no_tre.fit(durations=pxa_no_tre['survival_months'], 
                  event_observed=pxa_no_tre['death'], 
                  label='PXA without TRE')
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Plot the survival curves
    kmf_tre.plot_survival_function(ax=ax, show_censors=True, ci_show=True, color='blue')
    kmf_no_tre.plot_survival_function(ax=ax, show_censors=True, ci_show=True, color='orange')
    
    # Customize the plot
    plt.xlabel('Months')
    plt.ylabel('Survival Probability')
    plt.title('Kaplan-Meier Survival Curves')
    plt.xlim(0, 407)  # Set x-axis limit
    plt.xticks(range(0, 408, 50))  # Set x-ticks
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add at-risk counts if available in your lifelines version
    try:
        from lifelines.plotting import add_at_risk_counts
        add_at_risk_counts(kmf_tre, kmf_no_tre, ax=ax)
    except ImportError:
        logger.warning("Could not import add_at_risk_counts, skipping risk table")
    
    # Save the figure
    fig = plt.gcf()  # Get current figure
    save_figure(fig, "survival_analysis_tre_vs_no_tre", folder=output_dir)
    
    # Close the figure to free memory instead of showing it
    plt.close(fig)
    
    # Perform the log-rank test
    results = logrank_test(pxa_tre['survival_months'], 
                          pxa_no_tre['survival_months'], 
                          event_observed_A=pxa_tre['death'], 
                          event_observed_B=pxa_no_tre['death'])
    
    logger.info(f"Log-rank test p-value: {results.p_value:.4f}")
    
    return results

if __name__ == "__main__":
    main()


