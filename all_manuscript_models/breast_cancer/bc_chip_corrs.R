library(data.table)

D <- read.csv("/home/ubuntu/neural_ODE/all_manuscript_models/breast_cancer/all_inferred_influences_wide_FULL.csv")

D[is.na(D)] <- 0
D <- data.table(D)

print("correlations:")
cor(D$scale_to_500, D$h_cent)
cor(D$scale_to_2000, D$h_cent)
cor(D$scale_to_4000, D$h_cent)
cor(D$scale_to_11165, D$h_cent)

num_true_cent <- D[,sum(h_cent!=0)]

print("")
print("top10% sensitivity:")
D[order(-scale_to_500)][1:(500*0.10),][,sum(h_cent!=0)/num_true_cent]
D[order(-scale_to_2000)][1:(2000*0.10),][,sum(h_cent!=0)/num_true_cent]
D[order(-scale_to_4000)][1:(4000*0.10),][,sum(h_cent!=0)/num_true_cent]
D[order(-scale_to_11165)][1:(1117),][,sum(h_cent!=0)/num_true_cent]

print("")
print("top10% FNR:")
D[order(-scale_to_500)][(500*0.10):.N,][,sum(h_cent==0)/(11165-num_true_cent)]
D[order(-scale_to_2000)][(2000*0.10):.N,][,sum(h_cent==0)/(11165-num_true_cent)]
D[order(-scale_to_2000)][(4000*0.10):.N,][,sum(h_cent==0)/(11165-num_true_cent)]
D[order(-scale_to_11165)][1117:.N,][,sum(h_cent==0)/(11165-num_true_cent)]

print("")
print("top10% number missed:")
D[order(-scale_to_500)][(500*0.10):500,][,sum(h_cent!=0)]
D[order(-scale_to_2000)][(2000*0.10):2000,][,sum(h_cent!=0)]
D[order(-scale_to_2000)][(4000*0.10):4000,][,sum(h_cent!=0)]
D[order(-scale_to_11165)][1117:11165,][,sum(h_cent!=0)]

D[order(-scale_to_11165)][1117:11165,][h_cent!=0, as.character(gene)]
