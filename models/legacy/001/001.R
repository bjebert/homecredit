# CV: 0.743
# LB: 0.729

set.seed(0)
library(h2o)
h2o.init()

get_predictions <- function(train, test) {
    train[, TARGET := as.factor(TARGET)]
    test[, TARGET := as.factor(TARGET)]
    
    mdl <- h2o.gbm(training_frame = as.h2o(train),
                   y = "TARGET")
    
    predictions <- as.data.table(predict(mdl, as.h2o(test)))[["p1"]]
    
    return(predictions)
}