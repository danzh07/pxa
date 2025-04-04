# PXA Analysis Tool

## Overview

This tool provides comprehensive analysis of Pleomorphic Xanthoastrocytoma (PXA) clinical data with a focus on tumor-related epilepsy (TRE) and seizure outcomes. The analysis includes statistical comparisons of clinical characteristics, tumor features, and survival rates between patient groups.

## Features

- **Clinical Characteristics Analysis**: Compares demographics, tumor size, location, and molecular features between different patient groups.
- **Seizure Outcome Analysis**: Identifies factors associated with seizure freedom after tumor resection.
- **Survival Analysis**: Generates Kaplan-Meier survival curves comparing patients with and without tumor-related epilepsy.
- **Detailed Logging**: Creates comprehensive logs of all statistical analyses and findings.
- **Visualization**: Saves high-quality figures of survival analysis and other results.

## Installation

### Requirements

- Python 3.6+
- Required packages: numpy, pandas, matplotlib, scipy, lifelines, loguru

### Setup

1. Clone this repository to your local machine
2. Install required packages:
   ```bash
   pip install numpy pandas matplotlib scipy lifelines loguru
   ```

## Data Requirements

The analysis expects a CSV file named `pxa_data.csv` in the `data/` directory with the following key columns:

- `ID`: Patient identifier (used as index)
- `first_tumor_sz`: Whether seizure was the first symptom (1=yes, 0=no)
- `tumor_epilepsy`: Whether the patient developed tumor-related epilepsy (1=yes, 0=no)
- `presentation_1_age`: Age at presentation
- `tumor_size`: Size of tumor
- `tumor_mutational_burden`: Tumor mutational burden
- `sex_female`: Patient sex (1=female, 0=male)
- Various tumor location columns (`tumor_frontal`, `tumor_temp`, etc.)
- Mutation markers (`mut_BRAF_V600E`, `mut_CDKN2AB`, `mut_TERTp`)
- Treatment data (`resection_1_gross_total`, `add_chemo`, `add_radiation`)
- Outcome data (`sz-free_first_rxn`, `survival_months`, `death`)

## Usage

### Basic Usage

Run the analysis from the project root directory:

```bash
python run_pxa.py
```

### Specifying Output Directory

You can specify a custom output directory for logs and figures:

```python
from code.pxa import main
main(output_dir='path/to/output')
```

## Analysis Details

The tool performs three main analyses:

### 1. Clinical Characteristics Analysis (Table 1)
- Compares patients with initial seizure presentation vs. other presentations
- Compares patients with tumor-related epilepsy vs. without
- Analyzes continuous variables (age, tumor size, mutational burden)
- Analyzes categorical variables (sex, tumor locations, molecular features)

### 2. Seizure Outcome Analysis (Table 2)
- Evaluates seizure freedom rates after first resection
- Identifies factors associated with good seizure outcomes
- Analyzes both initial seizure patients and all tumor epilepsy patients

### 3. Survival Analysis
- Compares survival curves between TRE and non-TRE groups
- Performs log-rank test for statistical significance
- Generates Kaplan-Meier curves with confidence intervals

## Outputs

- **Log File**: Detailed analysis results saved to `pxa_analysis.log` in the output directory
- **Figures**: Survival analysis charts saved in PNG format
- **Console Output**: Summary of key findings displayed in the console

## Example Output

```
=== TABLE 1: CLINICAL CHARACTERISTICS ===
Total patients: 45
Initial seizure: 25 (0.56)
Initial other symptoms: 20 (0.44)
Tumor epilepsy: 32 (0.71)
No tumor epilepsy: 13 (0.29)

=== CONTINUOUS VARIABLES ===
Initial symptom: Age: Group 1 - Median: 22.0 (Range: 7.0-56.0)
Initial symptom: Age: Group 2 - Median: 20.5 (Range: 5.0-67.0)
Mann-Whitney U test for presentation_1_age: p-value=0.7801
...

=== SURVIVAL ANALYSIS ===
Log-rank test p-value: 0.0345
```

## Contributors

Daniel J. Zhou, Colin A. Ellis, Kevin Xie, Nishant Sinha, Sharon Xie, Kathryn A. Davis, Joel Stein, Tara Jennings, Stephen Bagley, Arati Desai, Patrick Y. Wen, David A. Reardon, Steven Tobochnik. Seizure characteristics and outcomes in patients with pleomorphic xanthoastrocytoma. (under review)



## Contact
Daniel Zhou, MD, MS  
Department of Neurology  
Hospital of the University of Pennsylvania  
3400 Spruce Street, 3 West Gates Building  
Philadelphia, PA 19104  
Email: daniel.zhou@pennmedicine.upenn.edu
