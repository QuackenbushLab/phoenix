library(data.table)
library(PRROC)



### will want genes to also have prior info so...
long_prior <- fread("/home/ubuntu/neural_ODE/mias_bcell_data/code/otter_clean_harmonized_full_prior.csv")
long_prior[,prior_pred := 1]
all_prior_affiliated_genes <- unique(c(long_prior$from, long_prior$to))
edges <- data.table(expand.grid(all_prior_affiliated_genes, all_prior_affiliated_genes))
setnames(edges, 
         old = c("Var1", "Var2"),
         new = c("from", "to"))

edges <- merge(edges, long_prior,
               by = c("from", "to"),
               all.x = T)
rm(long_prior)
gc()

long_val <- fread("/home/ubuntu/neural_ODE/mias_bcell_data/clean_data/otter_chip_val_clean.csv")
long_val[, val_pred := 1]
edges <- merge(edges, long_val[,.(from,to, val_pred)],
               by = c("from", "to"),
               all.x = T)
rm(long_val)
gc()

edges[is.na(val_pred), val_pred := 0]
edges[is.na(prior_pred), prior_pred := 0]


edges[, table(prior_pred, val_pred)]

PRROC_obj <- roc.curve(scores.class0 = edges$prior_pred,
                         weights.class0 = edges$val_pred,
                         curve=FALSE)
print(paste("AUC = ",
              round(PRROC_obj$auc, 3),
              ", num vals found =",
              edges[val_pred==1, .N]))
  
  

