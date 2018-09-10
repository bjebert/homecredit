# Hyperparams model -------------------------------------------------------

library(h2o)
h2o.init()

hyperparams <- data.table()
for(i in 1:nrow(res)) {
    hyperparams <- rbind(hyperparams, res[, params[[i]]])
}
hyperparams[, auc := res[["auc"]]]



mdl <- h2o.gbm(training_frame = as.h2o(hyperparams),
               y = "auc",
               max_depth = 2,
               ntrees = 50)

best_pred_auc <- 0
iter2 <- 0
h2o.no_progress()

while(TRUE) {
    iter2 <- iter2 + 1
    eta <- runif(1, min = 0.01, max = 0.25)
    gamma <- runif(1, min = 0, max = 10)
    max_depth <- sample(2:25, 1)
    colsample_bytree <- runif(1, min = 0.4, max = 1)
    colsample_bylevel <- runif(1, min = 0.4, max = 1)
    lambda <- runif(1, min = 0, max = 20)
    alpha <- runif(1, min = 0, max = 20)
    subsample <- runif(1, min = 0.4, max = 1)
    nrounds <- sample(1:500, 1)
    
    hyptest <- data.table(eta = eta,
                          gamma = gamma,
                          max_depth = max_depth,
                          colsample_bytree = colsample_bytree,
                          colsample_bylevel = colsample_bylevel,
                          lambda = lambda,
                          alpha = alpha,
                          subsample = subsample,
                          nrounds = nrounds)
    
    pred_auc <- as.data.table(h2o.predict(mdl, as.h2o(hyptest)))[["predict"]]
    
    if(pred_auc > best_pred_auc) {
        print(sprintf("New best PRED AUC: %f (Iteration %d)", pred_auc, iter2))
        best_pred_auc <- pred_auc
    }
}