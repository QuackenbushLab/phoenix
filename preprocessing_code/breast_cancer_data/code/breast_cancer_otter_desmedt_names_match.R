library(data.table)
library(stringr)

### clean otter tf names to align with desmedt
otter_motif_prior <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/cancer_breast_otter_motif.csv")
otter_tf_names <- data.table(tf = setdiff(names(otter_motif_prior),"Row"))

desmedt_gene_names <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/GeneNames.csv",
                            header = F)
counter <- 0
otter_gene_matcher <- function(this_name){
  counter <<- counter+1
  message(counter)
  my_regex <- paste0("^",this_name,"$|",
                     "^",this_name," ///|",
                     "^/// ",this_name)
  desm_names <- unique(grep(my_regex, desmedt_gene_names$V1, value = T)) 
  if(this_name %in% desm_names){
    desm_names <- c(this_name)
  }
  n_desm_names <- length(desm_names)
  names_str <- paste0(desm_names,  collapse = " , ")
  return(list(n_desm_names, names_str))
}

otter_tf_names[ ,c("n_desm_names","desm_names"):= otter_gene_matcher(tf),
                by = tf]

write.csv(otter_tf_names, "C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/otter_tf_names_desmedt_conv.csv",
          row.names = F)

## clean otter gene names to match with desmedt
otter_gene_names <- data.table(read.delim("C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/ensemble_id_to_symbol.txt",
                                          header = F))
setnames(otter_gene_names,
         old = c("V1","V2"),
         new = c("ensemble_id", "gene"))

otter_gene_names[ ,c("n_desm_names","desm_names"):= otter_gene_matcher(as.character(gene)),
                by = ensemble_id]

otter_gene_names[n_desm_names > 1, 
  desm_names := trimws((strsplit(desm_names, " , ")[[1]])[1]), 
  by = ensemble_id]

write.csv(otter_gene_names, "C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/otter_gene_names_desmedt_conv.csv",
          row.names = F)
