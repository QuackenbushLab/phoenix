library(data.table)

range01 <- function(x){(x-min(x))/(max(x)-min(x))}

num_tops <- 16
all_genes <- c(500, 2000, 4000, 11165)
all_top_genes <- c()

print(paste0("collecting top ", num_tops," genes from each set"))
for(this_gene in all_genes){
  this_gene_file <- paste0("/home/ubuntu/neural_ODE/all_manuscript_models/breast_cancer/inferred_influences/inferred_influence_",
                           this_gene,".csv")
  D <- fread(this_gene_file)
  gene_name_file <- paste0("/home/ubuntu/neural_ODE/all_manuscript_models/breast_cancer/desmedt_gene_names_", # nolint
                          this_gene,
                          ".csv")
  gene_names <- data.table(read.csv(gene_name_file))
  
  D <- cbind(gene_names, D)
  names(D) <- c("gene", "influence")
  D$gene <- as.character(D$gene)


  all_top_genes <- c(all_top_genes,
                     D[order(-influence),][1:num_tops, gene])
  
}

all_top_genes <- unique(all_top_genes)
print(paste0("Collected ", length(all_top_genes), " genes!"))

#print("Check if the union exits in all gene sets")
#for(this_gene in all_genes){
#  this_gene_file <- paste0("C:/STUDIES/RESEARCH/neural_ODE/all_manuscript_models/breast_cancer/harm_cents_",
#                           this_gene,".csv")
  
#  D <- fread(this_gene_file)
#  print(all(all_top_genes %in% D[, gene]))
#}


print("Now merging data for all these genes")
all_harms_merged <- data.table(gene = character(),
                               cent_rank = numeric(),
                               num_gene = numeric())
for(this_gene in all_genes){
  this_gene_file <- paste0("/home/ubuntu/neural_ODE/all_manuscript_models/breast_cancer/inferred_influences/inferred_influence_",
                           this_gene,".csv")
  
  D <- fread(this_gene_file)
  gene_name_file <- paste0("/home/ubuntu/neural_ODE/all_manuscript_models/breast_cancer/desmedt_gene_names_", # nolint
                          this_gene,
                          ".csv")
  gene_names <- data.table(read.csv(gene_name_file))
  
  D <- cbind(gene_names, D)
  names(D) <- c("gene", "influence")
  D$gene <- as.character(D$gene)


  D <-  D[order(-influence),]
  D[, cent_rank:= range01(influence)] #.I
  D_to_take <- D[gene %in% all_top_genes, .(gene, cent_rank)]
  D_to_take[, num_gene := paste0("scale_to_", this_gene)]
  all_harms_merged <- rbind(all_harms_merged, D_to_take)
}

all_harms_merged[ , .N , by = num_gene]

all_harms_wide <- dcast(all_harms_merged, 
                          gene ~ num_gene, value.var = "cent_rank")
setcolorder(all_harms_wide, 
            c("gene", 
              paste("scale_to", all_genes, sep = "_")))

true_harms <- fread("/home/ubuntu/neural_ODE/all_manuscript_models/breast_cancer/true_harm_cents_bc_out.csv")

all_harms_wide <- merge(all_harms_wide, 
                        true_harms, 
                        by = "gene", all.x = T)

all_harms_wide <- all_harms_wide[order(-scale_to_500,
                                       -scale_to_2000,
                                       -scale_to_4000,
                                       -scale_to_11165
                                       )]

print(all_harms_wide)

write.csv(all_harms_wide,
          "/home/ubuntu/neural_ODE/all_manuscript_models/breast_cancer/all_inferred_influences_wide_FULL.csv",
          row.names = F,
          na = "")
