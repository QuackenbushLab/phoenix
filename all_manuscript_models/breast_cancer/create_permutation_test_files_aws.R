library(data.table)
library(parallel)


simple_dot_prod <- function(i, this_path_binaries, this_phoenix_influence_scores){
  (as.matrix(this_path_binaries) %*% 
     this_phoenix_influence_scores[sample(.N, replace = F), 
                              influence])[1,1]
}

ecdf_fun <- function(x,perc){
  ecdf(x)(perc)
}
 
get_path_dot_product <- function(path_binaries, phoenix_influence_scores){
  n_run <<- n_run + 1
  setTxtProgressBar(pb, n_run)
  base_res <- (as.matrix(path_binaries) %*% #[, phoenix_influence_scores[, gene], with = F] 
    phoenix_influence_scores[, influence]) [1,1]
  all_perm_results <- parallel::mclapply(1:500,
                             simple_dot_prod,
                             this_path_binaries = path_binaries,
                             this_phoenix_influence_scores = phoenix_influence_scores,
                             mc.cores = 35)
  all_perm_results <- unlist(all_perm_results)                          
  mean_path_score <- mean(all_perm_results)
  sd_path_score <- sd(all_perm_results)
  phnx_z_score <- ifelse(sd_path_score > 0, 
                    (base_res - mean_path_score)/sd_path_score,
                    0)
  phnx_p_val <- mean(base_res < all_perm_results)

  return(list(phnx_z_score, 
              mean_path_score,
              sd_path_score, 
              phnx_p_val))
}


analysis_type <- "go_bp"
pathway_binary_file <- paste0("/home/ubuntu/neural_ODE/all_manuscript_models/breast_cancer/",
                              analysis_type,
                              "_pathway_binary_wide.csv")
path_DB <- fread(pathway_binary_file)

all_num_genes <- c(2000, 4000, 11165)
for(num_genes in all_num_genes){
  
  phnx_inf_file <- paste0("/home/ubuntu/neural_ODE/all_manuscript_models/breast_cancer/inferred_influences/inferred_influence_", # nolint
                          num_genes,
                          ".csv") 
  output_file <- paste0("/home/ubuntu/neural_ODE/all_manuscript_models/breast_cancer/",
                          analysis_type,
                          "_permtests/permtest_",
                           num_genes,
                            ".csv") 
  gene_name_file <- paste0("/home/ubuntu/neural_ODE/all_manuscript_models/breast_cancer/desmedt_gene_names_", # nolint
                          num_genes,
                          ".csv")
  phoenix_influence_scores <- fread(phnx_inf_file, header = FALSE)

  gene_names <- data.table(read.csv(gene_name_file))
  
  phoenix_influence_scores <- cbind(gene_names, phoenix_influence_scores)
  names(phoenix_influence_scores) <- c("gene", "influence")
  phoenix_influence_scores$gene <- as.character(phoenix_influence_scores$gene)

  phoenix_influence_scores_single_names <- phoenix_influence_scores[ !grepl("///", gene),]
  
  phoenix_influence_scores_multi_names <- unique(phoenix_influence_scores[ grepl("///", gene),
                                                                           .(gene_split = trimws(strsplit(gene,"///")[[1]]),
                                                                             influence = influence),
                                                                           by = gene])
  
  phoenix_influence_scores_multi_names[,gene := NULL]
  setnames(phoenix_influence_scores_multi_names, "gene_split", "gene")
  
  phoenix_influence_scores <- rbind(phoenix_influence_scores_single_names, 
                                    phoenix_influence_scores_multi_names)
  
  num_genes_multi_after_merge <- phoenix_influence_scores[, .N, by = gene][N>1, .N]
  print(paste("Number of genes appearing multiple times:", num_genes_multi_after_merge))
  print("Consolidating these genes using their averages")
  phoenix_influence_scores <- phoenix_influence_scores[, .(influence = mean(influence)), by = gene]
  
  phoenix_genes <- phoenix_influence_scores[, gene]
  print(paste("finally, we have:", length(phoenix_genes), "genes"))
  
  ### get relevant pathways (intersection)
  relevant_genes_from_phoenix <- phoenix_genes[phoenix_genes %in% names(path_DB)]
  num_phoenix_genes_in_paths <- length(relevant_genes_from_phoenix)
  print(paste("there are in fact",
              num_phoenix_genes_in_paths, 
              "genes that are found in some pathway.."))
  
  print("we will subset to these genes")
  this_path_DB <- path_DB[, c("pathway", relevant_genes_from_phoenix),
                          with = F]
  
  phoenix_influence_scores <- phoenix_influence_scores[gene %in% relevant_genes_from_phoenix]
  
  n_run <- 0
  num_pathways <- this_path_DB[, length(unique(pathway))]

  pb <- txtProgressBar(min = 0,      # Minimum value of the progress bar
                     max = num_pathways, # Maximum value of the progress bar
                     style = 3,    # Progress bar style (also available style = 1 and style = 2)
                     width = 50,   # Progress bar width. Defaults to getOption("width")
                     char = "=")   # Character used to create the bar

  X = this_path_DB[ ,get_path_dot_product(path_binaries = .SD, 
                                          phoenix_influence_scores = phoenix_influence_scores), 
                    by = pathway]
  names(X) <- c("pathway", "phnx_z_score", "mean_path_score", "sd_path_score", "phnx_p_val")
  X <- X[order(-phnx_z_score)]
  write.csv(X, output_file, row.names = F)

  print("")
  print(paste0("DONE! finished working on ",analysis_type, " for ", num_genes, " genes." ))
  print("_______________")
}