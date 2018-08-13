library(data.table)

applications <- fread("data/applications.csv", header = TRUE)

evaluate <- function(model_name) {
    train <- applications[origin == "train"]
    test <- applications[origin == "val"]
    
    predictions_env <- new.env()
    sys.source(sprintf("models/%s/%s.R", model_name, model_name), envir = predictions_env)
    
    predictions <- predictions_env$get_predictions(train, test)
    
    Metrics::auc(applications[origin == "val"][["TARGET"]], predictions)
}

evaluate("006")
