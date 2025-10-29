#' Download pretrained Mantis model weights
#'
#' Downloads the required .pt weight files from the official GitHub Releases
#' and saves them into a local `models/` directory so Mantis can load them.
#'
#' @param horizon Forecast horizon, either 4 or 8.
#' @param use_covariate Logical, whether to download covariate-enabled weights.
#' @param dest_dir Directory where to place the weights (default "models").
#' @export
mantis_download_weights <- function(horizon = 4L,
                                    use_covariate = TRUE,
                                    dest_dir = "models",
                                    version = "mantis-v1.0") {
  if (!horizon %in% c(4L, 8L)) {
    stop("horizon must be either 4 or 8", call. = FALSE)
  }
  if (!dir.exists(dest_dir)) {
    dir.create(dest_dir, recursive = TRUE)
  }
  
  suffix <- if (use_covariate) "cov" else "nocov"
  fname <- sprintf("mantis_%dw_%s.pt", horizon, suffix)
  
  base_url <- sprintf("https://github.com/carsondudley1/Mantis/releases/download/%s", version)
  url <- file.path(base_url, fname)
  
  dest_file <- file.path(dest_dir, fname)
  
  message("Downloading: ", url)
  utils::download.file(url, destfile = dest_file, mode = "wb")
  message("Saved model to: ", dest_file)
  invisible(dest_file)
}

