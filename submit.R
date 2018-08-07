# Submit a model

submit_predictions <- function(version, description = "", val_error = NA) {
    
    print("Reading files...")
    
    santander_train <- fread("data/train.csv", header = TRUE)
    santander_test <- fread("data/test.csv", header = TRUE)
    sample_submission <- fread("data/sample_submission.csv", header = TRUE)
    
    predictions_env <- new.env()
    sys.source(sprintf("models/%s/%s.R", version, version), envir = predictions_env)
    
    print("Getting predictions...")
    predictions <- predictions_env$get_predictions(santander_train, santander_test)
    
    if(length(predictions) != 49342) {
        stop("Length of predictions not correct")
    }
    
    if(!all.equal(santander_test[, ID], sample_submission[, ID])) {
        stop("IDs out of order between test and submission set")
    }
    
    print("Sanity checks complete...")
    
    sample_submission[, target := predictions]
    
    print("Writing predictions...")
    
    write_location <- sprintf("submissions/%s.csv", version)
    fwrite(sample_submission, write_location)
    
    print("Submitting to Kaggle...")
    
    system(sprintf("kaggle competitions submit -c santander-value-prediction-challenge -f %s -m '%s (CV: %s)'",
                   write_location, description, val_error))
    
    system(sprintf("kaggle competitions submit -c home-credit-default-risk -f %s -m '%s'",
                   write_location, description))
    
}

submit_predictions("004", "Rewrite love is in the air without blending in R", 1.666451)
