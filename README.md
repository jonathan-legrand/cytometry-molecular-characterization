# Molecular Characterization of AML from Flow Cytometry

Rapid molecular characterization is essential for risk stratification in Acute Myeloid Leukemia (AML). However, standard mutation assays often delay results beyond the window of early treatment decisions, particularly for patients in the gray zone between intensive and non-intensive chemotherapy. We hypothesize that flow cytometry data, routinely available within hours of patient admission, reflects molecular profile and can predict key mutations using machine learning.

This project implements three prediction methods :
1. **Baseline model**: Random forest using only basic clinical data (age, sex, blast percentage)
2. **Deep learning model**: Convolutional Neural Network (CNN) processing flow cytometry data (adaptation from Hu et al. 2020 - see train_network.py for reference)
3. **Multiple instance learning (MIL) model**: Each patient is represented as a bag of cells, where individual cells serve as instances, and the patient-level mutation status defines the bag label. A model is trained to predict the bag-level label from each instance individually. At inference time, bag-level predictions are obtained by aggregating instance-level predictions. This approach is usually referred to as single-instance learning in the MIL literature. In this study, decision trees were employed as classifiers due to their inherent interpretability.

This repository contains the preprocessing scripts, training pipelines, evaluation tools, and configuration files used for this study.

## Project Structure

### Main Scripts
- `preproc_all.R`: Preprocessing pipeline for raw flow cytometry FCS files (margin removal, compensation, transformation)
- `train_age_sex.py`: Trains baseline random forest model using demographic data
- `train_cell_level_model.py`: Trains cell-level classification model
- `train_network.py`: Trains multitube CNN model for end-to-end cytometry analysis
- `plot_results.py`: Generates comparison plots and statistical tests for model evaluation
- `test_on_external_set.py`: Evaluates final model on external test dataset

### Modules
- `flowcyt/`: Python package containing model architectures, data loading, training utilities, and evaluation metrics

### Configuration
- `config.yml`: Main configuration file specifying data paths and analysis parameters
- `env.yml`: Conda environment specification with Python dependencies
- `run_config/`: Directory containing JSON configuration files for different deep learning model runs
- `renv/`: R environment configuration for reproducible analysis

## Setup

### Python Environment
```bash
conda env create -f env.yml
conda activate flowcyt_subset
```

### R Environment
In R, restore the environment using:
```r
renv::restore()
```

## Workflow

### 1. Data Preprocessing
Preprocess raw FCS files:
- Modify data paths in `preproc_all.R` (set `data_dir`, `exp_suffix`, and `channel_suffix`)
- Run: `Rscript preproc_all.R`
- This step handles margin removal, compensation, and arcsinh transformation for both exploration and test sets

### 2. Configuration
Update `config.yml` with paths to preprocessed data:
- `FCS_PATH`: Path to preprocessed exploration set
- `TEST_FCS`: Path to preprocessed test set

### 3. Model Training
Train models in any order; the exploration set is split into train/validation (90/10) with 10-fold cross-validation:
```bash
python train_age_sex.py <target_col>
python train_cell_level_model.py <target_col> <n_cells>
python train_network.py <target_col> <config_name> <prefix>
```

Example: `python train_network.py NPM huaspossible final_run`

### 4. Model Comparison
Compare model performance using cross-validation results:
```bash
python plot_results.py <target_col>
```

Example: `python plot_results.py NPM`

This generates boxplots with statistical significance testing via permutation tests in the `plots` directory.

### 5. External Validation
Evaluate the best model on external test dataset:
```bash
python test_on_external_set.py <estimator_name> [n_seeds]
```

Arguments:
- `estimator_name`: Name of the trained model to evaluate (required)
- `n_seeds`: Number of resampling iterations for SD estimation (optional, default: 1)

To obtain uncertainty estimates with standard deviations, specify `n_seeds > 1`:
```bash
python test_on_external_set.py cell-level-model_predict-NPM_ncells-100_withtab-False_clf-decisiontreeclassifier_stamp-20260421_174225 10
```

This will perform cell resampling 10 times with different random seeds at inference, computing mean and SD estimates for AUROC, PPV, and NPV metrics across all resampling iterations.
