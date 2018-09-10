# R-squared feature engineering  (R2FE)

create_features <- function(data, target) {
    results <- data.table()
    numeric_cols <- colnames(data)[which(sapply(data, is.numeric))]
    numeric_cols <- numeric_cols[!numeric_cols %in% target]

    for(nc in numeric_cols) {
        feature_name <- nc
        
        x <- as.matrix(data[, feature_name, with = F])
        nas <- is.na(x[, 1])
        
        x <- as.matrix(x[!nas])
        y <- data[[target]][!nas]
                
        mdl <- lm.fit(x, y)
        results <- rbind(results, data.table(feature = feature_name, mean_res = mean(mdl$residuals^2), N = length(y)))
    }
    
    return(results[order(mean_res)])
}

z <- create_features(data, "TARGET")


# - -----------------------------------------------------------------------

z[1:10]

data_summary <- data[, .(TARGET = mean(TARGET), .N), by = .(GROUP = YEARS_BUILD_AVG)]

ggplot(data_summary, aes(x = GROUP, y = TARGET, weight = N)) +
    geom_point() +
    geom_smooth()

