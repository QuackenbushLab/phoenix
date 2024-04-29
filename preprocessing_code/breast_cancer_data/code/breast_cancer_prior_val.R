library(data.table)
library(stringr)
library(PRROC)



### will want genes to also have prior info so...
long_prior <- fread("/home/ubuntu/neural_ODE/breast_cancer_data/clean_data/otter_clean_harmonized_full_prior.csv")
long_prior[,prior_pred := 1]
all_prior_affiliated_genes <- unique(c(long_prior$from, long_prior$to))

long_val <- fread("/home/ubuntu/neural_ODE/breast_cancer_data/clean_data/otter_chip_val_clean.csv")
long_val[, val_pred := 1]

### start cleaning
full_data <- fread("/home/ubuntu/neural_ODE/breast_cancer_data/clean_data/Data_smooth_allgenes.csv")
gene_names <- fread("/home/ubuntu/neural_ODE/breast_cancer_data/clean_data/GeneNames.csv",
                    header = F)
names(gene_names) <- "gene"
full_data <- cbind(gene_names, full_data)

full_data <- full_data[!is.na(gene) & gene != "", ]

full_data <- full_data[,.SD[1], by = gene]

exprs_cols <- setdiff(names(full_data),"gene")
full_data_melt <- melt(full_data, 
                       id.vars = "gene",
                       measure.vars = exprs_cols,
                       variable.name = "time_point",
                       value.name = "exprs")

full_data_melt[, hist(exprs)]

high_var_genes <- full_data_melt[,var(exprs), by = gene][order(-V1)]
high_var_genes[, gene_in_prior:= gene %in% all_prior_affiliated_genes]
all_num_genes <- c(500, 2000, 4000, 11165)
#all_num_genes <- c(7000)
for (num_genes in all_num_genes){
  high_var_genes_this <- high_var_genes[gene_in_prior ==T, ][1:num_genes,gene]
  
  edges <- data.table(expand.grid(high_var_genes_this, high_var_genes_this))
  setnames(edges, 
           old = c("Var1", "Var2"),
           new = c("from", "to"))
  edges <- merge(edges, long_prior,
                 by = c("from", "to"),
                 all.x = T)
  #print("done merging")
 # edges[is.na(prior_pred), prior_pred := 0]
  
  edges <- merge(edges, long_val,
                 by = c("from", "to"),
                 all.x = T)
  #edges[is.na(val_pred), val_pred := 0]
  
  PRROC_obj <- roc.curve(scores.class0 = !is.na(edges$prior_pred),
                         weights.class0 = !is.na(edges$val_pred),
                         curve=FALSE)
  print(paste(num_genes,
              "genes, AUC = ",
              round(PRROC_obj$auc, 3),
              ", num vals found =",
              edges[val_pred==1, .N]))
  
  
}
