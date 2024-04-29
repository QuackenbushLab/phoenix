library(data.table)

range01 <- function(x){(x-min(x))/(max(x)-min(x))}

num_tops <- 11#basically all
all_genes <- c(500, 2000, 4000, 11165)
all_top_paths <- c()

analysis_type <- "reactome"

if (analysis_type == "reactome"){
  reactome_filter <- fread("/home/ubuntu/neural_ODE/all_manuscript_models/breast_cancer/reactome_child_to_parent.csv")
  reactome_filter[, child_descriptor := tolower(gsub("-"," ",child_descriptor))]
  reactome_filter[, child_descriptor := tolower(gsub(",","",child_descriptor))]
}



print(paste0("collecting top ", num_tops," pathways from each set"))
for(this_gene in all_genes){
  this_gene_file <- paste0("/home/ubuntu/neural_ODE/all_manuscript_models/breast_cancer/",
                           analysis_type,
                           "_permtests/permtest_",
                           this_gene,".csv")
  
  D <- fread(this_gene_file)
  if (analysis_type == "reactome"){
    D <- D[pathway %in% reactome_filter$child_descriptor]
  }
  #D <- D[phnx_z_score > 5, ]
  all_top_paths <- c(all_top_paths,
                     trimws(D[order(-phnx_z_score),][1:num_tops, pathway]))
  
}

all_top_paths <- unique(all_top_paths)
print(paste0("Collected ", length(all_top_paths), " paths!"))

#print("Check if the union exits in all gene sets")
#for(this_gene in all_genes){
#  this_gene_file <- paste0("/home/ubuntu/neural_ODE/all_manuscript_models/breast_cancer/harm_cents_",
#                           this_gene,".csv")
  
#  D <- fread(this_gene_file)
#  print(all(all_top_genes %in% D[, gene]))
#}


print("Now merging data for all these pathways")
all_permtest_merged <- data.table(pathway = character(),
                               z = numeric(),
                               mu_0 = numeric(), 
                               sd_0 = numeric(),
                               num_gene = numeric())
for(this_gene in all_genes){
  this_gene_file <- paste0("/home/ubuntu/neural_ODE/all_manuscript_models/breast_cancer/",
                           analysis_type,
                           "_permtests/permtest_",
                           this_gene,".csv")
  
  D <- fread(this_gene_file)
  D <-  D[order(-phnx_z_score),]
  D_to_take <- D[trimws(pathway) %in% all_top_paths, 
                 .(pathway = trimws(pathway),
                   z = round(phnx_z_score, 3),
                   mu_0 = round(mean_path_score,3),
                   sd_0 = round(sd_path_score,4))]
  D_to_take[, num_gene := paste0("scl2_", this_gene)]
  #D_to_take[, fold_permtest:= fold_permtest/this_gene] #.I
  #D_to_take[, fold_permtest:= range01(fold_permtest)] #.I
  
  all_permtest_merged <- rbind(all_permtest_merged, D_to_take)
}

all_permtest_merged <- all_permtest_merged[!(z == 0 & mu_0 == 0 & sd_0 == 0), ]
all_permtest_merged[ , .N , by = num_gene]


all_permtest_wide <- dcast(all_permtest_merged, 
                          pathway ~ num_gene, 
                          value.var = c("z", "mu_0", "sd_0"))


all_permtest_wide[,pathway := gsub(",",";", pathway)]
all_permtest_wide[,pathway := gsub("\\&","and", pathway)]


all_permtest_wide <- all_permtest_wide[order(-z_scl2_500,
                                       -z_scl2_2000,
                                       -z_scl2_4000,
                                       -z_scl2_11165
                                       )]
setcolorder(all_permtest_wide, 
            c("pathway", 
              paste(rep( c("z", "mu_0", "sd_0"), 4),
                    rep(paste("scl2", all_genes, sep = "_"), each = 3),
                    sep = "_"
              )
            )
)

#all_permtest_wide <- all_permtest_wide[order(pathway)]
merged_perm_tests <- fread(paste0("/home/ubuntu/neural_ODE/all_manuscript_models/breast_cancer/all_permtests_", analysis_type,"_wide.csv"))
all_permtest_wide$pathway  <- merged_perm_tests$pathway

print(all_permtest_wide)

write.table(all_permtest_wide,
          paste0("/home/ubuntu/neural_ODE/all_manuscript_models/breast_cancer/all_permtests_",
                 analysis_type,
                 "_supplement.txt"),
          sep = " & ",
          row.names = F,
          na = "",
          quote = F, 
          eol = "\\\\ \n")
