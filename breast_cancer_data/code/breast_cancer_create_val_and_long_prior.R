library(data.table)
library(stringr)

### Read in name files
otter_gene_names <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/otter_gene_names_desmedt_conv.csv")
otter_tf_names <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/otter_tf_names_desmedt_conv.csv")


### Clean validation chip (OTTER)
otter_val_chip <- data.table(read.delim("C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/otter_chipseq_postive_edges_breast.txt",
                                        header = F))
otter_val_chip <- merge(otter_val_chip, otter_tf_names,
                        by.x = "V1", by.y = "tf",)
setnames(otter_val_chip, "desm_names", "from")
otter_val_chip <- otter_val_chip[n_desm_names >0,]
otter_val_chip[, c("V1","n_desm_names"):= NULL]
otter_val_chip <- merge(otter_val_chip, otter_gene_names,
                        by.x = "V2", by.y = "ensemble_id")
otter_val_chip <- otter_val_chip[n_desm_names >0,]
setnames(otter_val_chip, "desm_names", "to")
otter_val_chip[, c("gene"):= NULL]

otter_val_chip <- unique(otter_val_chip)
write.csv(otter_val_chip, "C:/STUDIES/RESEARCH/neural_ODE/breast_cancer_data/clean_data/otter_chip_val_clean.csv",
          row.names = F)


#### Clean prior (OTTER)
otter_motif_prior <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/cancer_breast_otter_motif.csv")
otter_motif_prior <- merge(otter_motif_prior, otter_gene_names,
                        by.x = "Row", by.y = "ensemble_id")
otter_motif_prior <- otter_motif_prior[n_desm_names >0 & desm_names != "",]

otter_motif_prior <- melt(otter_motif_prior,
                          id.vars = c("desm_names"),
                          measure.vars = otter_tf_names$tf,
                          )
otter_motif_prior <- otter_motif_prior[value == 1,]
otter_motif_prior <- otter_motif_prior[, .(tf = variable, to = desm_names)]
otter_motif_prior <- merge(otter_motif_prior, otter_tf_names,
                           by = "tf")
otter_motif_prior <- otter_motif_prior[n_desm_names >0 & desm_names != "",]
otter_motif_prior[,c("n_desm_names","tf"):=NULL]
otter_motif_prior <- otter_motif_prior[,.(from = desm_names, to = to)]

otter_motif_prior <- unique(otter_motif_prior)

write.csv(otter_motif_prior, "C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/otter_clean_harmonized_full_prior.csv",
          row.names = F)

