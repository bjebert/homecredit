library(data.table)

applications <- fread("data/applications.csv")

train <- applications[origin == "train"]
val <- applications[origin == "val"]

source("models/tidy_xgb_796.R")
pred1 <- get_predictions(train, val)

source("models/baseline_lgbm_795.R")
pred2 <- get_predictions(train, val)

source("models/good_transformations_790.R")
pred3 <- get_predictions(train, val) 

target <- val[, TARGET]


# GLM Optimise -------------------------------------------------------

blend_data <- data.table()

for(i in 1:5000) {
    wt <- runif(3)
    wt <- wt / sum(wt)
    
    preds <- wt[1] * pred1 + wt[2] * pred2 + wt[3] * pred3
    blend_data <- rbind(blend_data, data.table(w1 = wt[1], w2 = wt[2], w3 = wt[3], auc = Metrics::auc(target, preds)))
}


p <- list(objective = "reg:linear",
          eval_metric = "rmse",
          eta = 0.05,
          max_depth = 6,
          min_child_weight = 30,
          gamma = 0,
          subsample = 0.85,
          colsample_bytree = 0.7,
          colsample_bylevel = 0.632,
          alpha = 0,
          lambda = 0,
          nrounds = 300)

dtrain <- xgb.DMatrix(data = as.matrix(blend_data[, -'auc']), label = blend_data[, auc])
m_xgb <- xgb.train(p, dtrain, p$nrounds)

samples <- matrix(runif(3*50000), ncol = 3)
samples <- t(apply(samples, 1, function(x) x / sum(x)))
dtest <- xgb.DMatrix(data = samples)

preds <- predict(m_xgb, dtest)
samples[which(preds == max(preds))[1], ]



# Submit ------------------------------------------------------------------


library(data.table)

applications <- fread("data/applications.csv")

train <- applications[origin == "train" | origin == "val"]
test <- applications[origin == "test"]

source("models/tidy_xgb_796.R")
pred1 <- get_predictions(train, test)

source("models/baseline_lgbm_795.R")
pred2 <- get_predictions(train, test)

source("models/good_transformations_790.R")
pred3 <- get_predictions(train, test) 

final1 <- 0.27345886 * pred1 + 0.4301827726 * pred2 + 0.2963584 * pred3
final2 <- 0.27773402  * pred1 + 0.4277969395  * pred2 + 0.2944690  * pred3
final3 <- 0.28280761  * pred1 + 0.4311042678  * pred2 + 0.2860881  * pred3
final4 <- 0.26946240  * pred1 + 0.4331031620   * pred2 + 0.2974344   * pred3
final5 <- 0.27345886 * pred1 + 0.4301827726 * pred2 + 0.2963584 * pred3