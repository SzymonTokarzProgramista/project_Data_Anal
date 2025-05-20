data {
  int<lower=1> N;             // liczba meczów
  int<lower=1> K;             // liczba graczy
  array[N] int<lower=1, upper=K> p1; // indeksy graczy (pierwszy zawodnik)
  array[N] int<lower=1, upper=K> p2; // indeksy graczy (drugi zawodnik)
  array[N] int<lower=0, upper=1> y;  // wynik meczu (1 jeśli p1 wygrał, 0 jeśli p2)
}

parameters {
  vector[K] skill; // umiejętności graczy (latent skill)
  real<lower=0> sigma; // odchylenie standardowe umiejętności graczy
}

model {
  // Priors
  skill ~ normal(0, sigma);
  sigma ~ cauchy(0, 2);

  // Likelihood
  for (n in 1:N)
    y[n] ~ bernoulli_logit(skill[p1[n]] - skill[p2[n]]);
}

generated quantities {
  vector[N] y_hat;    // przewidywania wyników
  vector[N] log_lik;  // logarytmy wiarygodności

  for (n in 1:N) {
    real p = inv_logit(skill[p1[n]] - skill[p2[n]]);
    y_hat[n] = bernoulli_rng(p);
    log_lik[n] = bernoulli_logit_lpmf(y[n] | skill[p1[n]] - skill[p2[n]]);
  }
}
