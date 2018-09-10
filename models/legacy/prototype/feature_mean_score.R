# Top features ------------------------------------------------------------



feature_mean_score <- function(res, cols) {
    fts <- data.table(Feature = cols, Total = 0, N = 0)
    
    for(i in 1:nrow(res)) {
        fts[Feature %in% res[i, x][[1]], Total := Total + res[i, auc]]
        fts[Feature %in% res[i, x][[1]], N := N + 1]
    }
    
    fts[, MeanAuc := Total / N]
    return(fts)    
}

res2 <- feature_mean_score(res, cols)
best_fts <- res2[order(-MeanAuc)][1:300][["Feature"]]