library(data.table)
library(stringr)


### will want genes to also have prior info so...
long_prior <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/mias_bcell_data/otter_clean_harmonized_full_prior.csv")
all_prior_affiliated_genes <- unique(c(long_prior$from, long_prior$to))

### start cleaning
full_data <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/mias_bcell_data/preprocessed_Bcell_GSE100441.csv")
full_data <- full_data[!is.na(gene) & gene != "", ]
full_data <- full_data[,.(expression = mean(expression)), by = .(ID, time, gene)]
setnames(full_data, 
         old = c("time", "expression"),
         new = c("time_point", "exprs"))

full_data[, exprs := log(1 + exprs)]
full_data[, hist(exprs, col = "salmon")] #we will normalize somehow
num_genes <-  length(all_prior_affiliated_genes) #basically all that the prior has
print(num_genes)
full_data <- full_data[gene %in% all_prior_affiliated_genes]
full_data[, length(unique(gene)), by = .(ID, time_point)]
time_vals <- unique(full_data$time_point)
time_cols <- paste0("t_", time_vals)

full_data <- dcast(full_data, ID + gene ~ time_point, value.var = "exprs")
names(full_data) <- c("sample", "gene", time_cols)

full_data <- full_data[order(sample, gene)] #this ordering step is important!
genes_names_to_save <- full_data[sample == "T1", .(gene)]

full_data <- full_data[, .SD[1:(.N+1)], 
                   by=sample][is.na(gene), (time_cols) := as.list(time_vals)]
full_data[, gene:= NULL]

full_data_treated <- full_data[sample %in% c("T1","T2")]
num_samples <- full_data_treated[, length(unique(sample))]
full_data_treated[, sample:= NULL]
top_row <- as.list(rep(NA, length(time_vals)))
top_row[[1]] <- num_genes
top_row[[2]] <- num_samples
full_data_treated <- rbind(top_row, full_data_treated)

full_data_control <- full_data[sample %in% c("U1","U2")]
num_samples <- full_data_control[, length(unique(sample))]
full_data_control[, sample:= NULL]
top_row <- as.list(rep(NA, length(time_vals)))
top_row[[1]] <- num_genes
top_row[[2]] <- num_samples
full_data_control <- rbind(top_row, full_data_control)


write.csv(genes_names_to_save , 
          "C:/STUDIES/RESEARCH/neural_ODE/mias_bcell_data/clean_data/mias_gene_names_14691.csv",
          row.names = F)

write.table(full_data_treated,
             "C:/STUDIES/RESEARCH/neural_ODE/mias_bcell_data/clean_data/mias_treated_14691genes_2samples_6T.csv", 
             sep=",",
             row.names = FALSE,
             col.names = FALSE,
             na = "")
write.table(full_data_control,
            "C:/STUDIES/RESEARCH/neural_ODE/mias_bcell_data/clean_data/mias_control_14691genes_2samples_6T.csv", 
            sep=",",
            row.names = FALSE,
            col.names = FALSE,
            na = "")

rm(full_data_treated)
rm(full_data_control)
rm(full_data)

## Now make relevant prior
edges <- long_prior
rm(long_prior)
genes <- genes_names_to_save
setnames(genes, "gene", "x")
genes[, gene_sl := .I]

edges <- merge(edges, genes, by.x = "from", by.y = "x")
edges[, from := gene_sl]
edges[, gene_sl:= NULL]

edges <- merge(edges, genes, by.x = "to", by.y = "x")
edges[, to := gene_sl]
edges[, gene_sl:= NULL]

edges[,activation := 1, 
      by = .(to, from)]

edge_mat <- matrix(0, nrow = nrow(genes), ncol = nrow(genes))

update_edge_mat <- function(edge_mat, from, to, activation){
  edge_mat[from, to] <<- activation #effect of row on column
}
edges[,
      update_edge_mat(edge_mat,from, to, activation), 
      by= .(from, to)]

#rm(edges)
write.table(edges[,.(from, to, activation)],
            "C:/STUDIES/RESEARCH/neural_ODE/mias_bcell_data/clean_data/edge_prior_matrix_mias_14691.csv", 
            sep = ",",
             row.names = F, 
          col.names = F)

#library(Matrix)
#B <- Matrix(edge_mat, sparse = TRUE)
#writeMM(B, "C:/STUDIES/RESEARCH/neural_ODE/mias_bcell_data/clean_data/edge_prior_matrix_mias_14691.csv")
#write.table(edge_mat,
#            "C:/STUDIES/RESEARCH/neural_ODE/breast_cancer_data/clean_data/edge_prior_matrix_desmedt_11165.csv",
#            sep = ",",
#            row.names = F, 
#            col.names = F
#)

