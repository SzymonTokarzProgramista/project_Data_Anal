data {
  int<lower=1> nGames;
  int<lower=2> nPlayers;
  array[nPlayers] real playerStrength;
  array[nGames] int<lower=1, upper=nPlayers> player1;
  array[nGames] int<lower=1, upper=nPlayers> player2;
  array[nGames] real p1Logit;
  array[nGames] real p2Logit;
  array[nGames] real oddsDiff;  // np. log(1/p1_odds) - log(1/p2_odds)
}

parameters {
  real<lower=0> sigma;
  real beta_rank;   // waga siły z rankingu ATP
  real beta_odds;   // waga kursów bukmacherskich
}

transformed parameters {
  vector[nGames] predictor;

  for (i in 1:nGames) {
    predictor[i] = beta_rank * (playerStrength[player1[i]] - playerStrength[player2[i]]) +
                   beta_odds * oddsDiff[i];
  }
}

model {
  sigma ~ normal(0, 1);
  beta_rank ~ normal(0, 1);
  beta_odds ~ normal(0, 1);

  p1Logit ~ normal(predictor, sigma);
  p2Logit ~ normal(-predictor, sigma);
}
