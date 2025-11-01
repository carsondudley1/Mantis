# ------------------------------------------------------------
# COVID-19 Death Forecasts with MANTIS
# ------------------------------------------------------------

# Install packages if not already installed
# install.packages("devtools")
# devtools::install("~/Desktop/rmantis")

library(rmantis)
library(ggplot2)

# ------------------------------------------------------------
# Setup Mantis environment
# ------------------------------------------------------------

mantis_setup(envname = "mantis-r", python_version = "3.10")

mantis_download_weights(
  horizon = 4,
  use_covariate = TRUE
)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

hosp_df <- read.csv("Data/covid_hospitalizations.csv")
deaths_df <- read.csv("Data/covid_deaths.csv")

state <- "mi"
hosp_ts <- hosp_df[[state]]   # covariate (hospitalizations)
deaths_ts <- deaths_df[[state]]  # target (deaths)

# ------------------------------------------------------------
# Generate forecasts
# ------------------------------------------------------------

start_weeks <- c(20, 25, 30, 35, 40, 45, 50, 55)
forecasts <- list()

for (start in start_weeks) {
  if (start + 4 > length(deaths_ts)) next
  
  input_target <- deaths_ts[1:start]
  input_cov <- hosp_ts[1:start]
  
  pred <- mantis_forecast(
    time_series = input_target,
    covariate   = input_cov,
    target_type = 2L,       # deaths
    covariate_type = 1L,    # hospitalizations
    horizon = 4L,
    use_covariate = TRUE
  )
  
  forecasts[[as.character(start)]] <- pred
}

# ------------------------------------------------------------
# Plot results
# ------------------------------------------------------------

history_weeks <- 20
first_week <- min(start_weeks)
start_plot <- max(1, first_week - history_weeks)
end_plot <- max(start_weeks) + 4

weeks <- start_plot:end_plot
truth_slice <- deaths_ts[start_plot:end_plot]

obs_df <- data.frame(
  week = weeks,
  deaths = truth_slice
)

p <- ggplot(obs_df, aes(x = week, y = deaths)) +
  geom_line(color = "black", size = 1.2) +
  labs(
    title = "Mantis Forecasts with Historical Context (Michigan)",
    x = "Week",
    y = "Weekly deaths"
  )

for (start in names(forecasts)) {
  fc <- forecasts[[start]]
  pred_weeks <- as.integer(start):(as.integer(start) + 3)
  
  fc_df <- data.frame(
    week = pred_weeks,
    median = fc[,5],
    lower = fc[,1],
    upper = fc[,9]
  )
  
  p <- p +
    geom_ribbon(data = fc_df, aes(x = week, ymin = lower, ymax = upper),
                fill = "#2f5aa8", alpha = 0.25, inherit.aes = FALSE) +
    geom_line(data = fc_df, aes(x = week, y = median),
              color = "#2f5aa8", size = 1, inherit.aes = FALSE)
}

p + theme_minimal()
