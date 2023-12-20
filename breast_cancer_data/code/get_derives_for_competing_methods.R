library(data.table)
D <- fread("C:/STUDIES/RESEARCH/neural_ODE/breast_cancer_data/clean_data/desmedt_11165genes_1sample_186T.csv")
top_row <- D[1, ]
time_row <- D[.N, ]
D <- D[-c(1, .N), ]

dim(D)
num_genes <- dim(D)[1]

### this part is just to see what's happening
idx = sample(1:500, 1)
mod_loess <- loess(as.numeric(D[idx, ]) ~ as.numeric(time_row), 
                   degree = 2, span = 2)
smooth_y <- predict(mod_loess)
plot(as.numeric(time_row), as.numeric(D[idx, ]), col = "dodgerblue", pch = 16)
points(as.numeric(time_row), predict(mod_loess), col = "red", type = "l", lwd = 2)
###############################


derivs <- matrix(NA, nrow = nrow(D), ncol = ncol(D))

for(idx in 1:num_genes){
  message(idx)
  mod_loess <- loess(as.numeric(D[idx, ]) ~ as.numeric(time_row), 
                     degree = 2, span = 2)
  smooth_y <- predict(mod_loess)
  d1 <- diff(smooth_y)/diff(as.numeric(time_row))  ## Simplest - first difference
  d1 <- c(d1[1],d1) #first TP has no backward difference, so repeat the 2nd one 
  derivs[idx, ] <- d1
}

derivs <- data.table(derivs)
derivs <- rbind(top_row, derivs, time_row)


write.table(derivs,
            "C:/STUDIES/RESEARCH/neural_ODE/breast_cancer_data/clean_data/desmedt_11165genes_1sample_186T_DERIVATIVES.csv", 
            sep=",",
            row.names = FALSE,
            col.names = FALSE,
            na = "")
