# Kernel info

set.seed(0)
library(data.table)
library(xgboost)
library(lightgbm)

source("~/utilities/col_convert.R")
source("~/utilities/one_hot_encode.R")
source("~/utilities/target_encode.R")
source("~/utilities/remove_categorical.R")

# Initial read ------------------------------------------------------------

applications <- fread("data/applications.csv", header = TRUE)
train <- applications[origin == "train"]
test <- applications[origin == "val"]

bureau <- fread("data/bureau.csv")
bureau_balance <- fread("data/bureau_balance.csv")
credit_card_balance <- fread("data/credit_card_balance.csv")
installments <- fread("data/installments_payments.csv")
pos_cash_balance <- fread("data/POS_CASH_balance.csv")
previous_application <- fread("data/previous_application.csv")

# Preprocess data ---------------------------------------------------------


Mode <- function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
}


merge_bureau <- function(data) {
    
    bureau_balance_summary <- bureau_balance[, .(MONTHS_BALANCE_MIN = min(MONTHS_BALANCE),
                                                 MONTHS_BALANCE_MAX = max(MONTHS_BALANCE),
                                                 MONTHS_BALANCE_COUNT = .N),
                                             by = "SK_ID_BUREAU"]
    
    bureau_sum <- merge(bureau, bureau_balance_summary, by = "SK_ID_BUREAU", all.x = T)
    
    # BUREAU 
    bureau_summary <- bureau_sum[, .(BUREAU_HAD_BAD_DEBT = (sum(CREDIT_ACTIVE == "Bad debt") > 0),
                                     BUREAU_PRIMARY_CURRENCY = Mode(CREDIT_CURRENCY),
                                     BUREAU_MOST_RECENT_CREDIT_APPLICATION = min(DAYS_CREDIT, na.rm = T),
                                     BUREAU_AVERAGE_DAYS_OVERDUE = mean(CREDIT_DAY_OVERDUE),
                                     BUREAU_COUNT_OVERDUE = (sum(CREDIT_DAY_OVERDUE) > 0),
                                     BUREAU_MAX_OVERDUE= max(AMT_CREDIT_MAX_OVERDUE),
                                     BUREAU_SUM_OVERDUE = sum(AMT_CREDIT_SUM_OVERDUE),
                                     BUREAU_MEAN_PCT_OVERDUE = mean(AMT_CREDIT_SUM_OVERDUE > 0),
                                     BUREAU_DAYS_CREDIT_MEAN = mean(DAYS_CREDIT),
                                     BUREAU_DAYS_CREDIT_SD = sd(DAYS_CREDIT),
                                     BUREAU_DAYS_CREDIT_ENDDATE_MEAN = mean(DAYS_CREDIT_ENDDATE),
                                     BUREAU_DAYS_CREDIT_UPDATE_MEAN = mean(DAYS_CREDIT_UPDATE),
                                     BUREAU_MEAN_MAX_OVERDUE = mean(AMT_CREDIT_MAX_OVERDUE),
                                     BUREAU_AMT_CREDIT_SUM_MEAN = mean(AMT_CREDIT_SUM),
                                     BUREAU_AMT_CREDIT_TOTAL = sum(AMT_CREDIT_SUM),
                                     BUREAU_MEAN_OVERDUE = mean(AMT_CREDIT_SUM_OVERDUE),
                                     BUREAU_MEAN_CREDIT_LIMIT = mean(AMT_CREDIT_SUM_LIMIT),
                                     BUREAU_TOTAL_CREDIT_LIMIT = sum(AMT_CREDIT_SUM_LIMIT),
                                     BUREAU_ANNUITY_MAX = max(AMT_ANNUITY),
                                     BUREAU_ANNUITY_MEAN = mean(AMT_ANNUITY),
                                     BUREAU_SUM_CREDIT_PROLONG = sum(CNT_CREDIT_PROLONG),
                                     BUREAU_MIN_MONTHS_BALANCE = min(MONTHS_BALANCE_MIN),
                                     BUREAU_MAX_MONTHS_BALANCE = max(MONTHS_BALANCE_MAX),
                                     BUREAU_MONTHS_BALANCE_MEAN = mean(MONTHS_BALANCE_COUNT),
                                     BUREAU_MONTHS_BALANCE_TOTAL = sum(MONTHS_BALANCE_COUNT),
                                     N = .N), by = SK_ID_CURR]
    
    bureau_active <- bureau_sum[CREDIT_ACTIVE == "Active", .(BUREAU_TOTAL_ACTIVE_CREDIT = sum(AMT_CREDIT_SUM, na.rm = T),
                                                             BUREAU_TOTAL_ACTIVE_DEBT = sum(AMT_CREDIT_SUM_DEBT, na.rm = T)),
                                by = SK_ID_CURR]
    
    bureau_closed <- bureau_sum[CREDIT_ACTIVE == "Closed", .(BUREAU_TOTAL_CLOSED_CREDIT = sum(AMT_CREDIT_SUM, na.rm = T),
                                                             BUREAU_TOTAL_CLOSED_DEBT = sum(AMT_CREDIT_SUM_DEBT, na.rm = T)),
                                by = SK_ID_CURR]
    
    data <- merge(data, bureau_summary, by = "SK_ID_CURR", all.x = TRUE, sort = FALSE)
    data <- merge(data, bureau_active, by = "SK_ID_CURR", all.x = TRUE, sort = FALSE)
    data <- merge(data, bureau_closed, by = "SK_ID_CURR", all.x = TRUE, sort = FALSE)
    
    data[, BUREAU_PRIMARY_CURRENCY_CURRENCY_3 := NULL]
    
    return(data)
}

merge_previous_applications <- function(data) {
    previous_application[DAYS_FIRST_DRAWING == 365243, DAYS_FIRST_DRAWING := NA]
    previous_application[DAYS_FIRST_DUE == 365243, DAYS_FIRST_DUE := NA]
    previous_application[DAYS_LAST_DUE_1ST_VERSION == 365243, DAYS_LAST_DUE_1ST_VERSION := NA]
    previous_application[DAYS_LAST_DUE == 365243, DAYS_LAST_DUE := NA]
    previous_application[DAYS_TERMINATION == 365243, DAYS_TERMINATION := NA]
    
    previous_application[, APP_CREDIT_PERC := AMT_APPLICATION / AMT_CREDIT]
    
    previous_approved <- previous_application[NAME_CONTRACT_STATUS == "Approved",
                                              .(PA_APPROVED_AMT_ANNUITY_MAX = max(AMT_ANNUITY, na.rm = T),
                                                PA_APPROVED_AMT_ANNUITY_MEAN = mean(AMT_ANNUITY, na.rm = T),
                                                PA_APPROVED_AMT_APPLICATION_MAX = max(AMT_APPLICATION, na.rm = T),
                                                PA_APPROVED_AMT_APPLICATION_MEAN = mean(AMT_APPLICATION, na.rm = T),
                                                PA_APPROVED_AMT_CREDIT_MAX = max(AMT_CREDIT, na.rm = T),
                                                PA_APPROVED_AMT_CREDIT_MEAN = mean(AMT_CREDIT, na.rm = T),
                                                PA_APPROVED_APP_CREDIT_PERC_MAX = max(APP_CREDIT_PERC, na.rm = T),
                                                PA_APPROVED_APP_CREDIT_PERC_MEAN = mean(APP_CREDIT_PERC, na.rm = T),
                                                PA_APPROVED_AMT_DOWN_PAYMENT_MAX = max(AMT_DOWN_PAYMENT, na.rm = T),
                                                PA_APPROVED_AMT_DOWN_PAYMENT_MEAN = mean(AMT_DOWN_PAYMENT, na.rm = T),
                                                PA_APPROVED_AMT_GOODS_PRICE_MAX = max(AMT_GOODS_PRICE, na.rm = T),
                                                PA_APPROVED_AMT_GOODS_PRICE_MEAN = mean(AMT_GOODS_PRICE, na.rm = T),
                                                PA_APPROVED_HOUR_APPR_PROCESS_START_MED = median(HOUR_APPR_PROCESS_START, na.rm = T),
                                                PA_APPROVED_RATE_DOWN_PAYMENT_MAX = max(RATE_DOWN_PAYMENT, na.rm = T),
                                                PA_APPROVED_RATE_DOWN_PAYMENT_MEAN = mean(RATE_DOWN_PAYMENT, na.rm = T),
                                                PA_APPROVED_DAYS_DECISION_MAX = max(DAYS_DECISION, na.rm = T),
                                                PA_APPROVED_DAYS_DECISION_MEAN = mean(DAYS_DECISION, na.rm = T),
                                                PA_APPROVED_CNT_PAYMENT_MAX = max(CNT_PAYMENT, na.rm = T),
                                                PA_APPROVED_CNT_PAYMENT_MEAN = mean(CNT_PAYMENT, na.rm = T)),
                                              by = SK_ID_CURR]
    
    previous_refused <- previous_application[NAME_CONTRACT_STATUS == "Refused",
                                             .(PA_REFUSED_AMT_ANNUITY_MAX = max(AMT_ANNUITY, na.rm = T),
                                               PA_REFUSED_AMT_ANNUITY_MEAN = mean(AMT_ANNUITY, na.rm = T),
                                               PA_REFUSED_AMT_APPLICATION_MAX = max(AMT_APPLICATION, na.rm = T),
                                               PA_REFUSED_AMT_APPLICATION_MEAN = mean(AMT_APPLICATION, na.rm = T),
                                               PA_REFUSED_AMT_CREDIT_MAX = max(AMT_CREDIT, na.rm = T),
                                               PA_REFUSED_AMT_CREDIT_MEAN = mean(AMT_CREDIT, na.rm = T),
                                               PA_REFUSED_APP_CREDIT_PERC_MAX = max(APP_CREDIT_PERC, na.rm = T),
                                               PA_REFUSED_APP_CREDIT_PERC_MEAN = mean(APP_CREDIT_PERC, na.rm = T),
                                               PA_REFUSED_AMT_DOWN_PAYMENT_MAX = max(AMT_DOWN_PAYMENT, na.rm = T),
                                               PA_REFUSED_AMT_DOWN_PAYMENT_MEAN = mean(AMT_DOWN_PAYMENT, na.rm = T),
                                               PA_REFUSED_AMT_GOODS_PRICE_MAX = max(AMT_GOODS_PRICE, na.rm = T),
                                               PA_REFUSED_AMT_GOODS_PRICE_MEAN = mean(AMT_GOODS_PRICE, na.rm = T),
                                               PA_REFUSED_HOUR_APPR_PROCESS_START_MED = median(HOUR_APPR_PROCESS_START, na.rm = T),
                                               PA_REFUSED_RATE_DOWN_PAYMENT_MAX = max(RATE_DOWN_PAYMENT, na.rm = T),
                                               PA_REFUSED_RATE_DOWN_PAYMENT_MEAN = mean(RATE_DOWN_PAYMENT, na.rm = T),
                                               PA_REFUSED_DAYS_DECISION_MAX = max(DAYS_DECISION, na.rm = T),
                                               PA_REFUSED_DAYS_DECISION_MEAN = mean(DAYS_DECISION, na.rm = T),
                                               PA_REFUSED_CNT_PAYMENT_MAX = max(CNT_PAYMENT, na.rm = T),
                                               PA_REFUSED_CNT_PAYMENT_MEAN = mean(CNT_PAYMENT, na.rm = T)),
                                             by = SK_ID_CURR]
    
    previous_agg <- previous_application[,
                                         .(PA_AMT_ANNUITY_MAX = max(AMT_ANNUITY, na.rm = T),
                                           PA_AMT_ANNUITY_MEAN = mean(AMT_ANNUITY, na.rm = T),
                                           PA_AMT_APPLICATION_MAX = max(AMT_APPLICATION, na.rm = T),
                                           PA_AMT_APPLICATION_MEAN = mean(AMT_APPLICATION, na.rm = T),
                                           PA_AMT_CREDIT_MAX = max(AMT_CREDIT, na.rm = T),
                                           PA_AMT_CREDIT_MEAN = mean(AMT_CREDIT, na.rm = T),
                                           PA_APP_CREDIT_PERC_MAX = max(APP_CREDIT_PERC, na.rm = T),
                                           PA_APP_CREDIT_PERC_MEAN = mean(APP_CREDIT_PERC, na.rm = T),
                                           PA_AMT_DOWN_PAYMENT_MAX = max(AMT_DOWN_PAYMENT, na.rm = T),
                                           PA_AMT_DOWN_PAYMENT_MEAN = mean(AMT_DOWN_PAYMENT, na.rm = T),
                                           PA_AMT_GOODS_PRICE_MAX = max(AMT_GOODS_PRICE, na.rm = T),
                                           PA_AMT_GOODS_PRICE_MEAN = mean(AMT_GOODS_PRICE, na.rm = T),
                                           PA_HOUR_APPR_PROCESS_START_MED = as.numeric(median(HOUR_APPR_PROCESS_START, na.rm = T)),
                                           PA_RATE_DOWN_PAYMENT_MAX = max(RATE_DOWN_PAYMENT, na.rm = T),
                                           PA_RATE_DOWN_PAYMENT_MEAN = mean(RATE_DOWN_PAYMENT, na.rm = T),
                                           PA_DAYS_DECISION_MAX = max(DAYS_DECISION, na.rm = T),
                                           PA_DAYS_DECISION_MEAN = mean(DAYS_DECISION, na.rm = T),
                                           PA_CNT_PAYMENT_MAX = max(CNT_PAYMENT, na.rm = T),
                                           PA_CNT_PAYMENT_MEAN = mean(CNT_PAYMENT, na.rm = T),
                                           PA_NUMBER_APPROVED = sum(NAME_CONTRACT_STATUS == "Approved"),
                                           PA_NUMBER_REFUSED = sum(NAME_CONTRACT_STATUS == "Refused")),
                                         by = SK_ID_CURR]
    
    previous_agg[, PA_APPROVE_RATE := PA_NUMBER_APPROVED / (PA_NUMBER_APPROVED + PA_NUMBER_REFUSED)]
    
    previous_agg <- merge(previous_agg, previous_approved, by = "SK_ID_CURR", all.x = T)
    previous_agg <- merge(previous_agg, previous_refused, by = "SK_ID_CURR", all.x = T)
    
    data <- merge(data, previous_agg, by = "SK_ID_CURR", all.x = T)
    
    return(data)
}

merge_pos_cash <- function(data) {
    
    pos_summary <- pos_cash_balance[, .(POS_MONTHS_MAX = max(MONTHS_BALANCE),
                                        POS_MONTHS_MEAN = mean(MONTHS_BALANCE),
                                        POS_SK_DPD_MAX = max(SK_DPD),
                                        POS_SK_DPD_MEAN = mean(SK_DPD),
                                        POS_SK_DPD_DEF_MAX = max(SK_DPD_DEF),
                                        POS_SK_DPD_DEF_MEAN = mean(SK_DPD_DEF),
                                        POS_MEAN_ACTIVE_STATUS = mean(NAME_CONTRACT_STATUS == "Active"),
                                        POS_MEAN_COMPLETED_STATUS = mean(NAME_CONTRACT_STATUS == "Completed"),
                                        POS_COUNT = .N),
                                    by = SK_ID_CURR]
    
    data <- merge(data, pos_summary, by = "SK_ID_CURR", all.x = T)
    
    return(data)
}

merge_installments <- function(data) {
    
    installments[, PAYMENT_PERC := AMT_PAYMENT / AMT_INSTALMENT]
    installments[, PAYMENT_DIFF := AMT_INSTALMENT - AMT_PAYMENT]
    
    ins_summary <- installments[, .(INS_UNDERPAY_RATIO = mean(AMT_PAYMENT < AMT_INSTALMENT),
                                    INS_UNDERPAY_COUNT = sum(AMT_PAYMENT < AMT_INSTALMENT),
                                    INS_PAY_RATIO = mean(AMT_PAYMENT / AMT_INSTALMENT),
                                    INS_PAY_COUNT = .N,
                                    INS_LATE_PAYMENTS = sum(DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT),
                                    INS_MEAN_INSTALMENT = mean(AMT_INSTALMENT),
                                    INS_SUM_INSTALMENTS = sum(AMT_INSTALMENT),
                                    INS_UNIQUE_INSTALMENTS = uniqueN(NUM_INSTALMENT_VERSION),
                                    INS_MEAN_PERC = mean(PAYMENT_PERC),
                                    INS_MIN_PERC = min(PAYMENT_PERC),
                                    INS_SD_PERC = sd(PAYMENT_PERC),
                                    INS_MAX_PERC = max(PAYMENT_PERC),
                                    INS_MEAN_DIFF = mean(PAYMENT_DIFF),
                                    INS_MIN_DIFF = min(PAYMENT_DIFF),
                                    INS_SD_DIFF = sd(PAYMENT_DIFF),
                                    INS_MAX_DIFF = max(PAYMENT_DIFF)), 
                                by = SK_ID_CURR]
    
    ins_late <- installments[DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT, 
                             .(INS_MEAN_DAYS_LATE = mean(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT),
                               INS_MEDIAN_DAYS_LATE = median(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)),
                             by = SK_ID_CURR]
    
    ins_summary <- merge(ins_summary, ins_late, all.x = TRUE)
    ins_summary[, INS_LATE_PAY_RATIO := INS_LATE_PAYMENTS / INS_PAY_COUNT]
    
    data <- merge(data, ins_summary, by = "SK_ID_CURR", all.x = TRUE, sort = FALSE)
    
    return(data)
}


merge_credit_card_balance <- function(data) {
    cc <- copy(credit_card_balance)
    cc[, SK_ID_PREV := NULL]
    numeric_cols <- colnames(cc)[which(sapply(cc, is.numeric))]
    
    cc_mean <- cc[, lapply(.SD, mean, na.rm = T), .SDcols = numeric_cols, by = SK_ID_CURR]
    cc_max <- cc[, lapply(.SD, max, na.rm = T), .SDcols = numeric_cols, by = SK_ID_CURR]
    cc_sum <- cc[, lapply(.SD, sum, na.rm = T), .SDcols = numeric_cols, by = SK_ID_CURR]
    
    cc_mean[, 1 := NULL]
    cc_max[, 1:2 := NULL]
    cc_sum[, 1:2 := NULL]
    
    colnames(cc_mean)[2:ncol(cc_mean)] <- sprintf("CC_%s_MEAN", colnames(cc_mean)[2:ncol(cc_mean)])
    colnames(cc_max) <- sprintf("CC_%s_MAX", colnames(cc_max))
    colnames(cc_sum) <- sprintf("CC_%s_SUM", colnames(cc_sum))
    
    cc_combined <- cbind(cc_sum, cbind(cc_mean, cc_max))
    
    data <- merge(data, cc_combined, by = "SK_ID_CURR", all.x = TRUE, sort = FALSE)
    
    return(data)
}


preprocess_data <- function(data) {
    data <- col_convert(data, "integer", "numeric")  # Convert integer columns to numeric
    
    # Convert 2-valued character columns to boolean
    data[, IS_CONTRACT_CASH := NAME_CONTRACT_TYPE == "Cash loans"]
    data[, IS_MALE := CODE_GENDER == "M"]
    data[, IS_OWN_CAR := FLAG_OWN_CAR == "Y"]
    data[, IS_OWN_REALTY := FLAG_OWN_REALTY == "Y"]
    data[, IS_EMERGENCY_MODE := ifelse(data[["EMERGENCYSTATE_MODE"]] == "Yes", 
                                       TRUE, 
                                       ifelse(data[["EMERGENCYSTATE_MODE"]] == "No", 
                                              FALSE, 
                                              NA))]
    
    data[DAYS_EMPLOYED == 365243, DAYS_EMPLOYED := NA]
    
    # One-hot encode cols
    
    data[NAME_TYPE_SUITE == "", NAME_TYPE_SUITE := "NA"]
    data[OCCUPATION_TYPE == "", OCCUPATION_TYPE := "NA"]
    data[FONDKAPREMONT_MODE == "", FONDKAPREMONT_MODE := "NA"]
    data[HOUSETYPE_MODE == "", HOUSETYPE_MODE := "NA"]
    data[WALLSMATERIAL_MODE == "", WALLSMATERIAL_MODE := "NA"]
    
    data[, DOWNPAYMENT := AMT_CREDIT - AMT_GOODS_PRICE]
    data[, DOWNPAYMENT_PCT := DOWNPAYMENT / AMT_GOODS_PRICE]
    data[, LOAN_LENGTH_YEARS := AMT_CREDIT / AMT_ANNUITY]
    data[, CREDIT_TO_GOODS := AMT_CREDIT / AMT_GOODS_PRICE]
    
    data[, INCOME_PER_CHILD := AMT_INCOME_TOTAL / (1 + CNT_CHILDREN)]
    data[, CHILDREN_RATIO := CNT_CHILDREN / CNT_FAM_MEMBERS]
    
    data <- merge(data, data[, .(INCOME_ORG_MEDIAN = median(AMT_INCOME_TOTAL)), by = ORGANIZATION_TYPE], by = "ORGANIZATION_TYPE", all.x = T)
    
    data[, EMPLOYMENT_TO_AGE_RATIO := DAYS_EMPLOYED / DAYS_BIRTH]
    data[, EMPLOYMENT_TO_AGE_SQRT := sqrt(DAYS_EMPLOYED / DAYS_BIRTH)]
    data[, ANNUITY_TO_INCOME := AMT_ANNUITY / (1 + AMT_INCOME_TOTAL)]
    
    data[, EXT_SOURCES_PRODUCT := EXT_SOURCE_1 * EXT_SOURCE_2 * EXT_SOURCE_3]
    data[, EXT_SOURCES_MEAN := (EXT_SOURCE_1 + EXT_SOURCE_2 + EXT_SOURCE_3) / 3]
    data[, EXT_SOURCES_SD := apply(data[, .(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)], 1, sd)]
    data[is.na(EXT_SOURCES_SD), EXT_SOURCES_SD := mean(data[["EXT_SOURCES_SD"]], na.rm = T)]
    
    data[, CAR_TO_BIRTH_RATIO := OWN_CAR_AGE / DAYS_BIRTH]
    data[, CAR_TO_EMPLOYMENT_RATIO := OWN_CAR_AGE / DAYS_EMPLOYED]
    
    data[, PHONE_TO_BIRTH_RATIO := DAYS_LAST_PHONE_CHANGE / DAYS_BIRTH]
    data[, PHONE_TO_EMPLOYMENT_RATIO := DAYS_LAST_PHONE_CHANGE / DAYS_EMPLOYED]
    
    data[, CREDIT_TO_INCOME_RATIO := AMT_CREDIT / AMT_INCOME_TOTAL]
    
    drop_columns <- sprintf("FLAG_DOCUMENT_%s", c(2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21))
    data[, (drop_columns) := NULL]
    
    data <- merge_bureau(data)
    data <- merge_previous_applications(data)
    data <- merge_pos_cash(data)
    data <- merge_installments(data)
    data <- merge_credit_card_balance(data)
    
    x <- colnames(data)
    
    # Remove columns from data
    x <- x[!(x %in% c("origin", "fold", "SK_ID_CURR", "CODE_GENDER"))]
    data <- data[, x, with = FALSE]
    
    data <- one_hot_encode(data)
    
    colnames(data) <- gsub(" ", "_", colnames(data))
    colnames(data) <- gsub(",", "", colnames(data))
    colnames(data) <- gsub(":", "", colnames(data))
    colnames(data) <- gsub("/", "", colnames(data))
    colnames(data) <- toupper(colnames(data))
    
    return(data)
}

train_processed <- preprocess_data(train)
test_processed <- preprocess_data(test)

train_processed <- col_convert(train_processed, "character", "")
test_processed <- col_convert(test_processed, "character", "")

train_processed[, TARGET := NULL]
test_processed[, TARGET := NULL]  # Shouldn't be here, but just in case

fwrite(train_processed, "data/train_processed.csv", row.names = F)
fwrite(test_processed, "data/test_processed.csv", row.names = F)