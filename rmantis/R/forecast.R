#' Forecast with Mantis
#'
#' @param time_series Numeric vector (required).
#' @param covariate Optional numeric vector.
#' @param target_type Integer (0=cases, 1=hosp, 2=deaths).
#' @param covariate_type Integer or NULL.
#' @param horizon Integer forecast horizon (4 or 8).
#' @param use_covariate Logical, default TRUE.
#' @param envname Conda environment (default "mantis-r").
#'
#' @return A numeric matrix of forecasts.
#' @export
mantis_forecast <- function(time_series,
                            covariate = NULL,
                            target_type = 2L,
                            covariate_type = NULL,
                            horizon = 4L,
                            use_covariate = TRUE,
                            envname = "mantis-r") {
  reticulate::use_condaenv(envname, required = TRUE)
  mantis <- reticulate::import("mantis")
  model <- mantis$Mantis(forecast_horizon = as.integer(horizon),
                         use_covariate = use_covariate)
  preds <- model$predict(
    time_series = as.integer(time_series),
    covariate   = if (!is.null(covariate)) as.integer(covariate) else NULL,
    target_type = as.integer(target_type),
    covariate_type = covariate_type
  )
  return(as.matrix(preds))
}
