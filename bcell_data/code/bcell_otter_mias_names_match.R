library(data.table)
library(stringr)

### clean otter tf names to align with miasedt
otter_motif_prior <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/cancer_breast_otter_motif.csv")
otter_tf_names <- data.table(tf = setdiff(names(otter_motif_prior),"Row"))

mias_gene_names <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/mias_bcell_data/gene_names_mias.csv",
                            header = T)


counter <- 0
otter_gene_matcher <- function(this_name){
  counter <<- counter+1
  if (counter%%100 ==0){message(counter)}
  
  my_regex <- paste0("^",this_name,"$|",
                     "^",this_name," ///|",
                     "^/// ",this_name)
  mias_names <- unique(grep(my_regex, mias_gene_names$gene, value = T)) 
  if(this_name %in% mias_names){
    mias_names <- c(this_name)
  }
  n_mias_names <- length(mias_names)
  names_str <- paste0(mias_names,  collapse = " , ")
  return(list(n_mias_names, names_str))
}

otter_tf_names[ ,c("n_mias_names","mias_names"):= otter_gene_matcher(tf),
                by = tf]

write.csv(otter_tf_names, "C:/STUDIES/RESEARCH/ODE_project_local_old/mias_bcell_data/otter_tf_names_mias_conv.csv",
          row.names = F)

## clean otter gene names to match with miasedt
otter_gene_names <- data.table(read.delim("C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/ensemble_id_to_symbol.txt",
                                          header = F))
setnames(otter_gene_names,
         old = c("V1","V2"),
         new = c("ensemble_id", "gene"))

counter <- 0
otter_gene_names[ ,c("n_mias_names","mias_names"):= otter_gene_matcher(as.character(gene)),
                by = ensemble_id]

otter_gene_names[n_mias_names > 1, 
  mias_names := trimws((strsplit(mias_names, " , ")[[1]])[1]), 
  by = ensemble_id]

write.csv(otter_gene_names, "C:/STUDIES/RESEARCH/ODE_project_local_old/mias_bcell_data/otter_gene_names_mias_conv.csv",
          row.names = F)
