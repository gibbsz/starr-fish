// The input data is a dataframe 'y' of dim 'N, 2'.
data {
  int<lower=0> N; // number of cells
  int<lower=0> E; // number of Negative Control Elements
  array[N, E] int x; // observed Negative Control transcripts
  vector<lower=0>[N] xx; // observed total RNA transcripts
  vector<lower=0>[E] L; // observed library size for negative controls
  real<lower=0> mean_x_mean; // estimated mean_x for background distribution
  real<lower=0> mean_x_std; // estimated mean_x std
  real<lower=0> beta_x_mean; // estimated dispersion
  real<lower=0> beta_x_std; // estimated dispersion std
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'beta'.

parameters {
  // real<lower=0> fold_x; // Previous declaration
  real log_fold_x; // Declare the log of fold_x
  real<lower=0> mean_x;
  real<lower=0> beta_x;
}

transformed parameters {
  real<lower=0> fold_x = exp(log_fold_x); // Transform back to positive scale
}

model {
  //// Priors
  // Prior for fold_x centered at 1 (log(1)=0)
  // The standard deviation of 1.0 is weakly informative,
  // allowing the data to have a strong say.
  log_fold_x ~ normal(0, 1);

  mean_x ~ normal(mean_x_mean, mean_x_std);
  beta_x ~ normal(beta_x_mean, beta_x_std);

  //// Main program
  for (j in 1:E) {
    // The calculation of mu now uses the transformed fold_x
    vector[N] mu = mean_x * L[j] * xx * beta_x * fold_x;
    vector[N] phi = rep_vector(beta_x, N);
    target += neg_binomial_lpmf(x[, j] | mu, phi);
  }
}
