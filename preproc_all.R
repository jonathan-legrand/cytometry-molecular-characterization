# Preprocessing Script for Cytometry Data
#
# This script performs preprocessing on cytometry FCS files, including margin removal,
# compensation for spillover, and transformation. It processes all FCS files in the
# specified raw data directory and outputs preprocessed files to a new directory.
#
# Key steps:
# 1. Load FCS files from the 'Raw' subdirectory
# 2. Remove margins using PeacoQC
# 3. Apply compensation based on spillover matrix
# 4. Transform fluorescence channels using arcsinh transformation
# 5. Save preprocessed files to the 'preprocessed_{exp_suffix}' directory
#
# Usage: Run this script in an R environment with the required libraries installed.
#        Modify the constants section to match your data directory and naming conventions.
#
# Author: Jonathan Legrand

library(flowCore)
library(fs)
library(PeacoQC)
library(glue)

get_spillover_matrix <- function(obj) {
  if (!is.null(obj$SPILL)) {
    return(obj$SPILL)
  } else if (!is.null(obj$`$SPILLOVER`)) {
    return(obj$`$SPILLOVER`)
  } else {
    stop("No spillover matrix found in the object.")
  }
}

# Constants definition: Adapt to your environment and data
## This is the path to the repo containing a "Raw" repository
## which should itself contain all the fcs files you want to
## preprocess 
data_dir <- "/home/jonalegr/FCS_CPaleari"
exp_suffix <- "finalrepo" # The way you want to call that preproc run

## Depending on the dataset, pnn names do not always follow the
## same convention
## In our data, it's " INT" for the exploration
## set and "-A" for the test set
channel_suffix <- " INT"

# Data loading
dir_raw <- fs::path(data_dir, "Raw")

dir_prepr <- fs::path(data_dir, glue("preprocessed_{exp_suffix}"))
dir.create(dir_prepr)

# Log code for reproducibility
file.copy(
  fs::path(getwd(), "preproc_all.R"),
  fs::path(data_dir, glue("preproc_{exp_suffix}.R"))
)

# Load files
file_pattern <- ".*\\.fcs"
files <- list.files(path = dir_raw, pattern = file_pattern)

channels_of_interest <- c(glue("FS{channel_suffix}"), glue("SS{channel_suffix}"), paste0("FL", 1:10, glue("{channel_suffix}")))
fluorochromes <- channels_of_interest[3:length(channels_of_interest)]
asinh <- arcsinhTransform(b = 1/5)
translist <- transformList(fluorochromes, asinh)

for (file in files) {
  print(glue("Processing {file}"))
  ff <- read.FCS(fs::path(dir_raw, file), truncate_max_range = FALSE)
  keyword(ff)$"@SAMPLEID1" <- NULL # Anonymisation sanity check
  ff_m <- PeacoQC::RemoveMargins(ff, channels_of_interest)

  comp_mat <- get_spillover_matrix(spillover(ff_m))
  colnames(comp_mat) <- fluorochromes
  ff_c <- flowCore::compensate(ff_m, comp_mat)

  ff_t <- flowCore::transform(ff_c, translist)

  write.FCS(ff_t, fs::path(dir_prepr, file))
  
}

print(glue("Output stored in {dir_prepr}"))
