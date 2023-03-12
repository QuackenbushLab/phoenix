library(data.table)
library(stringr)


### will want genes to also have prior info so...
long_prior <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/otter_clean_harmonized_full_prior.csv")
all_prior_affiliated_genes <- unique(c(long_prior$from, long_prior$to))
### start cleaning

full_data <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/Data_smooth_allgenes.csv")
gene_names <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/GeneNames.csv",
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
num_genes <- 11165 #basically all that the prior has
high_var_genes <- full_data_melt[,var(exprs), by = gene][order(-V1)]
high_var_genes[, gene_in_prior:= gene %in% all_prior_affiliated_genes]
high_var_genes <- high_var_genes[gene_in_prior ==T, ][1:num_genes,gene]

full_data <- full_data[gene %in% high_var_genes]
full_data <- full_data[order(gene)]
genes_names_to_save <- full_data[, .(gene)]

full_data[, gene := NULL]
full_data <- full_data/15 #NORMALIZE to 0-1

time_vals <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/LatantTime.csv")

full_data <- rbind(full_data, time_vals)
top_row <- as.list(rep(NA, dim(time_vals)[2]))
top_row[[1]] <- num_genes
top_row[[2]] <- 1
full_data <- rbind(top_row, full_data)

write.csv(genes_names_to_save , 
          "C:/STUDIES/RESEARCH/neural_ODE/breast_cancer_data/clean_data/desmedt_gene_names_11165.csv",
          row.names = F)

write.table(full_data,
             "C:/STUDIES/RESEARCH/neural_ODE/breast_cancer_data/clean_data/desmedt_11165genes_1sample_186T.csv", 
             sep=",",
             row.names = FALSE,
             col.names = FALSE,
             na = "")

## Now make relevant prior
edges <- long_prior
genes <- genes_names_to_save
setnames(genes, "gene", "x")
genes[, gene_sl := .I]

edges <- merge(edges, genes, by.x = "from", by.y = "x")
edges[, from := gene_sl]
edges[, gene_sl:= NULL]

edges <- merge(edges, genes, by.x = "to", by.y = "x")
edges[, to := gene_sl]
edges[, gene_sl:= NULL]

edges[,activation := sample(c(-0.5, 0.5),1), 
      by = .(to, from)]

edge_mat <- matrix(0, nrow = nrow(genes), ncol = nrow(genes))

update_edge_mat <- function(edge_mat, from, to, activation){
  edge_mat[from, to] <<- activation #effect of row on column
}
edges[,
      update_edge_mat(edge_mat,from, to, activation), 
      by= .(from, to)]

library(Matrix)
B <- Matrix(edge_mat, sparse = TRUE)
writeMM(B, "C:/STUDIES/RESEARCH/neural_ODE/breast_cancer_data/clean_data/edge_prior_matrix_desmedt_11165.csv")
#write.table(edge_mat,
#            "C:/STUDIES/RESEARCH/neural_ODE/breast_cancer_data/clean_data/edge_prior_matrix_desmedt_11165.csv",
#            sep = ",",
#            row.names = F, 
#            col.names = F
#)

