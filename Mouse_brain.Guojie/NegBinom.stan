// The input data is a dataframe 'y' of dim 'N, 2'.
data {
  int<lower=0> N; // number of cells
  int<lower=0> E; // number of Negative Control Elements
  array[N, E] int x; // observed Negative Control transcripts
  vector<lower=0>[N] xx; // observed total RNA transcripts
  vector<lower=0>[E] L; // observed library size for negative controls
}

transformed data {
  // Precompute the inverses to avoid repeated division in the loop.
  vector[N] inv_xx = 1.0 ./ xx;
}
// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'beta'.
parameters {
  real<lower=0> mean_x; // Negative Binomial Mean, the activity mean
  real<lower=0> beta_x; // estimated dispersion 
}

model {
  ////Main program
  //Loop through data points
  ////
  for (j in 1:E) {
    vector[N] mu = mean_x * L[j] * xx * beta_x;
    vector[N] phi = rep_vector(beta_x, N);
    target += neg_binomial_lpmf(x[, j] | mu, phi);
  }
}

