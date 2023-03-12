library(data.table)
library(PRROC)

edges <- fread("C:/Users/Intekhab Hossain/Desktop/model_inspect_chalmers_690_prior_50_0411/edge_properties.csv")
edges <- edges[, .(from, to, activation)]

genes <- fread("C:/Users/Intekhab Hossain/Desktop/model_inspect_chalmers_690_prior_50_0411/gene_names.csv")
genes[ ,x:= gsub("_input","",x)]
genes[, gene_sl := .I]

num_genes <- genes[,.N]

edges <- merge(edges, genes, by.x = "from", by.y = "x")
edges[, from := gene_sl]
edges[, gene_sl:= NULL]

edges <- merge(edges, genes, by.x = "to", by.y = "x")
edges[, to := gene_sl]
edges[, gene_sl:= NULL]

edges[, true_edge := 1]

all_pos_edges <- data.table(expand.grid(1:num_genes, 1:num_genes))
names(all_pos_edges) <- c("to", "from")
all_pos_edges <- merge(all_pos_edges, 
                       edges[,.(to, from, true_edge)], 
                       by = c("to","from"),
                       all.x = T)
edges[, true_edge := NULL]

noise <- 0.05
noise_perc <- noise/0.5
edges[,activation := as.integer(2*(activation - 0.5))]
edges[, edge_to_flip:= 0]
edges[, edge_to_reloc := 0]
edges[sample(.N,.N*noise_perc,replace = F) , edge_to_flip := 1]
edges[sample(.N,.N*noise_perc,replace = F) , edge_to_reloc := 1]

reloc_edge <- function(to, from, activation){
  #change edge location
  return(as.list(
    c(
      as.integer(sample(1:num_genes,2, replace = F)),
      as.integer(activation))
      )
  )
}

#MUTATE edges
edges[edge_to_reloc ==1, 
      c("to", "from", "activation"):= 
        reloc_edge(to, from, activation),
      by = .(to, from, activation)]
#edges[edge_to_flip == 1, activation := -1*activation]
edges[, prior_edge:= 1]

all_pos_edges <- merge(all_pos_edges, 
                       edges[,.(to, from, prior_edge)], 
                       by = c("to","from"),
                       all.x = T)
edges[, prior_edge := NULL]

PRROC_obj <- roc.curve(scores.class0 = !is.na(all_pos_edges$prior_edge),
                       weights.class0 = !is.na(all_pos_edges$true_edge),
                       curve=FALSE)

print(paste(noise,
            "noise, AUC = ",
            PRROC_obj$auc)
            )


edge_mat <- matrix(0, nrow = nrow(genes), ncol = nrow(genes))

update_edge_mat <- function(edge_mat, from, to, activation){
  edge_mat[from, to] <<- activation #effect of row on column
}
edges[,
      update_edge_mat(edge_mat,from, to, activation), 
      by= .(from, to)]

prior_name <- paste0("edge_prior_matrix_chalmers_690_noise_",noise,".csv")
write.table(edge_mat,
          paste0("C:/STUDIES/RESEARCH/neural_ODE/ground_truth_simulator/clean_data/",
                 prior_name),
          sep = ",",
          row.names = F, 
          col.names = F
          )
