#!/usr/bin/env Rscript

# Load required libraries
library(cmdstanr)
library(readr)
library(parallel)
library(foreach)
library(doParallel)

# first infer background
rna <- read_csv("rna.csv", col_names = TRUE)[, -1]  # observed RNA transcripts xx
df <- read_csv("df.csv", col_names = TRUE, name_repair = "minimal")[, -1]  # observed Negative Control transcripts x
lib <- read_csv("lib.csv", col_names = TRUE)[, -1]  # library size L
# df add a column of all negative control CREs sum
df$sum <- rowSums(df)
lib <- rbind(lib, data.frame(counts = colSums(lib)))
N <- nrow(df)  # number of cells
E <- ncol(df)  # number of Negative Control Elements
stan_data <- list(
  N = N,
  E = E,
  x = as.matrix(df),  # observed Negative Control transcripts
  xx = as.vector(rna[[1]]),  # observed total RNA transcripts
  L = as.vector(lib[[1]])   # library size for negative controls
)
# Compile the Stan model
model <- cmdstan_model("NegBinom.stan")
# Run MCMC sampling
fit <- model$sample(
  data = stan_data,
  chains = 1,
  parallel_chains = 1,
  iter_warmup = 1000,
  iter_sampling = 2000,
  refresh = 500
)
res <- list()
res$background <- fit$summary()

# Now run for all non-Negative Control Elements
# Read input data
df <- read_csv("df_all.csv", col_names = TRUE, name_repair = "minimal")[, -1]  # observed Negative Control transcripts x
lib <- read_csv("lib_all.csv", col_names = TRUE)[, -1]  # library size L
# Prepare data for Stan
N <- nrow(df)  # number of cells

# Setup parallel backend
n_cores <- 96  # Use all cores except one
cl <- makeCluster(n_cores)
registerDoParallel(cl)

# Parallel execution of Stan models
results_parallel <- foreach(i = 1:dim(df)[2]) %dopar% {
  source('~/Pipeline/Rsession.init.R')
  library(cmdstanr)
  set_cmdstan_path('/share/vault/Users/gz2294/miniconda3/envs/r4-base/bin/cmdstan')
  stan_data <- list(
    N = N,
    E = 1,
    x = as.matrix(df[,i]),  # observed Negative Control transcripts
    xx = as.vector(rna[[1]]),  # observed total RNA transcripts
    L = as.vector(lib[[1]][i]),   # library size for negative controls
    mean_x_mean = res$background$mean[2],
    beta_x_mean = res$background$mean[3],
    mean_x_std = res$background$sd[2],
    beta_x_std = res$background$sd[3]
  )
  # Compile the Stan model
  model <- cmdstan_model("NegBinom2.stan")
  # Run MCMC sampling
  fit <- model$sample(
    data = stan_data,
    chains = 1,
    parallel_chains = 1,
    iter_warmup = 1000,
    iter_sampling = 2000,
    refresh = 500
  )
  # Return summary with column name
  list(name = colnames(df)[i], summary = fit$summary())
}

# Stop parallel backend
stopCluster(cl)

# Convert results back to named list format
for (result in results_parallel) {
  res[[result$name]] <- result$summary
}
# Save results
saveRDS(res, file = 'fit.RDS')

# check res
fdc <- c(); ess_bulk <- c(); sig <- c()
for (i in colnames(df)) {
  ess_bulk <- c(ess_bulk, res[[i]]$ess_bulk[5])
  if (res[[i]]$ess_bulk[2] < 1000) {
    fdc <- c(fdc, NA)
    sig <- c(sig, FALSE)
  } else {
    fdc <- c(fdc, res[[i]]$mean[5])
    sig <- c(sig, res[[i]]$q5[5] > 1)
  }
}
atac <- read.csv('atac.csv', row.names = 1)
fdc2 <- colSums(df) / lib
cor.test(fdc, fdc2$counts)
plot(fdc, fdc2$counts)
colSums(df)[colSums(df) < 1 & fdc > 10]
