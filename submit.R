# Submit a model

submit_predictions <- function(version, description = "", val_error = NA, dry_run = FALSE) {
    
    print("Reading files...")
    
    applications <- fread("data/applications.csv", header = TRUE)
    sample_submission <- fread("data/sample_submission.csv", header = TRUE)
    
    predictions_env <- new.env()
    sys.source(sprintf("models/%s/%s.R", version, version), envir = predictions_env)
    
    print("Getting predictions...")
    predictions <- predictions_env$get_predictions(
        applications[origin %in% c("train", "val")], applications[origin == "test"]
    )
    
    if(length(predictions) != 48744) {
        stop("Length of predictions not correct")
    }
    
    if(!all.equal(applications[origin == "test", SK_ID_CURR], sample_submission[, SK_ID_CURR])) {
        stop("IDs out of order between test and submission set")
    }
    
    print("Sanity checks complete...")
    
    sample_submission[, TARGET := predictions]
    
    if(dry_run) {
        return(sample_submission)
    }
    
    print("Writing predictions...")
    
    write_location <- sprintf("submissions/%s.csv", version)
    fwrite(sample_submission, write_location)
    
    print("Submitting to Kaggle...")

    system(sprintf("kaggle competitions submit -c home-credit-default-risk -f %s -m '%s (CV: %s)'",
                   write_location, description, val_error))
    
}

preds <- submit_predictions("001", "Test run", 1.666451, dry_run = TRUE)
