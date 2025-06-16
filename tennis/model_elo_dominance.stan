data {
  int<lower=1> nGames;
  int<lower=2> nPlayers;
  array[nPlayers] real playerStrength;
  array[nGames] int<lower=1, upper=nPlayers> player1;
  array[nGames] int<lower=1, upper=nPlayers> player2;
  array[nGames] real p1Logit;
  array[nGames] real p2Logit;
  array[nGames] real oddsDiff;
  array[nGames] real dominance;  // Nowa zmienna – znormalizowana dominacja
}

parameters {
  real<lower=0> sigma;
  real beta_rank;
  real beta_odds;
  real beta_dom;  // Nowy współczynnik dla dominacji
}

transformed parameters {
  vector[nGames] predictor;

  for (i in 1:nGames) {
    predictor[i] = beta_rank * (playerStrength[player1[i]] - playerStrength[player2[i]]) +
                   beta_odds * oddsDiff[i] +
                   beta_dom * dominance[i];  // Dodajemy wpływ dominacji
  }
}

model {
  sigma ~ normal(0, 1);
  beta_rank ~ normal(0, 1);
  beta_odds ~ normal(0, 1);
  beta_dom ~ normal(0, 1);  // Priorytet dominacji

  p1Logit ~ normal(predictor, sigma);
  p2Logit ~ normal(-predictor, sigma);
}

generated quantities {
  vector[nGames] log_lik;
  for (i in 1:nGames) {
    log_lik[i] = normal_lpdf(p1Logit[i] | predictor[i], sigma);
  }
}
