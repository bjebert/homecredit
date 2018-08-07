# Data builder

library(data.table)

application_train <- fread("data/application_train.csv")
application_test <- fread("data/application_test.csv")
bureau <- fread("data/bureau.csv")
bureau_balance <- fread("data/bureau_balance.csv")
credit_card_balance <- fread("data/credit_card_balance.csv")
installments <- fread("data/installments_payments.csv")
pos_cash_balance <- fread("data/POS_CASH_balance.csv")
previous_application <- fread("data/previous_application.csv")
sample_submission <- fread("data/sample_submission.csv")


# Validation settings ----------------------------------------------------------------

application_test[, TARGET := NA]
application_train[, origin := "train"]
application_test[, origin := "test"]

applications <- rbind(application_train, application_test)

rm(application_train)
rm(application_test)

applications[origin == "train", fold := 1:.N %% 10]
applications[fold %in% 7:9, origin := "val"]

fwrite(applications, "data/applications.csv", row.names = FALSE)
