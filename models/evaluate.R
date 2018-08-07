applications <- fread("data/applications.csv", header = TRUE)

evaluate <- function(version) {
    predictions_env <- new.env()
    sys.source(sprintf("models/%s/%s.R", version, version), envir = predictions_env)
    
    predictions <- predictions_env$get_predictions(applications[origin == "train"], applications[origin == "val"])
    
    Metrics::auc(applications[origin == "val"][["TARGET"]], predictions)
}

evaluate("001")

