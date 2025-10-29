#' Setup rmantis (installs reticulate + Python Mantis)
#'
#' This ensures reticulate is installed, then sets up a conda env
#' with PyTorch + Mantis (reusing existing Miniconda if present).
#'
#' @param envname Name of the conda environment (default "mantis-r").
#' @param python_version Python version to use (default "3.10").
#' @export
mantis_setup <- function(envname = "mantis-r", python_version = "3.10") {
  # Ensure reticulate is installed
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    install.packages("reticulate")
  }
  library(reticulate)
  
  # Check if miniconda path already exists
  mc_path <- reticulate::miniconda_path()
  if (!dir.exists(mc_path)) {
    message("Installing Miniconda...")
    reticulate::install_miniconda()
  } else {
    message("Using existing Miniconda at: ", mc_path)
  }
  
  # Create env if missing
  envs <- tryCatch(reticulate::conda_list()$name, error = function(e) character())
  if (!(envname %in% envs)) {
    message("Creating conda environment: ", envname)
    reticulate::conda_create(envname, python_version = python_version)
  } else {
    message("Conda environment '", envname, "' already exists.")
  }
  
  # Activate and install packages
  use_condaenv(envname, required = TRUE)
  message("Installing PyTorch + Mantis into env: ", envname)
  
  reticulate::conda_install(envname, c("torch", "torchvision", "torchaudio"), pip = TRUE)
  reticulate::py_run_string("
import sys, subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'git+https://github.com/carsondudley1/Mantis.git'])
")
  message("Setup complete. Ready to use Mantis.")
}



