# FlowCyt Subset

This is a smaller, reproducible subset of the flow cytometry project.

## Files

- `preproc_all.R`: Preprocessing script for flow cytometry data.
- `train_network.py`: Script to train the neural network model.
- `train_age_sex.py`: Script to train age and sex prediction model.
- `train_cell_level_model.py`: Script to train cell-level model.
- `plot_results.py`: Script to plot results (converted from notebook).
- `test_on_external_set.py`: Script to test on external dataset.
- `flowcyt/`: Python module with necessary files.
- `flowcytr/`: R module.
- `env.yml`: Conda environment with minimal Python dependencies.
- `renv/`: R environment (can be stripped further).
- `config.yml`: Configuration file.

## Setup

1. Create conda environment: `conda env create -f env.yml`
2. Activate: `conda activate flowcyt_subset`
3. For R, use renv: `renv::restore()` in R.
4. Use the values you need in the R script

## Usage

Once 
1. First, we need preprocessed files. Run preprocessing on the exploration set, and optionnally, on the test set. To do so, modify the required constants in the R script and run it using `Rscript preproc_all.R`
2. The path to preprocessed files should be replaced in `FCS_PATH` for the exploration set and `TEST_FCS` for the test set.
3. 

Run the scripts as needed. Adjust paths in config.yml if necessary.