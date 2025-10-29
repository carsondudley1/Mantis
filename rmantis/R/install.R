#' Install Python + Mantis
#'
#' This sets up a conda environment with PyTorch and the Mantis package.
#' @param envname Name of the conda environment (default "mantis-r").
#' @param python_version Python version to use (default "3.10").
#' @export
mantis_install <- function(envname = "mantis-r", python_version = "3.10") {
  reticulate::install_miniconda()
  reticulate::conda_create(envname, python_version = python_version)
  reticulate::conda_install(envname, c("torch", "torchvision", "torchaudio"), pip = TRUE)
  reticulate::py_run_string("
import sys, subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/carsondudley1/Mantis.git'])
")
  message('Mantis installed in environment: ', envname)
}
