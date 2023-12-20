


setwd('~/Projects/PHOENIX_data/GSE100441_RAW/')

timepoints <- c('00', '01', '02', '04', '07', '15')

data_U1 <- do.call(cbind, lapply(1:length(timepoints), function(i)
{
  timepoint <- timepoints[i]
  id <- 2683114 + i
  dat <- read.table(paste0('GSM',id,'_U',timepoint,'_RTX_genes.fpkm_tracking'), header=T, sep='\t')
  return(dat$FPKM)
}
))
colnames(data_U1) <- timepoints
rownames(data_U1) <- read.table(paste0('GSM2683115_U00_RTX_genes.fpkm_tracking'), header=T, sep='\t')$gene_id

data_U2 <- do.call(cbind, lapply(1:length(timepoints), function(i)
{
  timepoint <- timepoints[i]
  id <- 2683126 + i
  dat <- read.table(paste0('GSM',id,'_RU',timepoint,'_RTX_genes.fpkm_tracking'), header=T, sep='\t')
  return(dat$FPKM)
}
))
colnames(data_U2) <- timepoints
rownames(data_U2) <- read.table(paste0('GSM2683127_RU00_RTX_genes.fpkm_tracking'), header=T, sep='\t')$gene_id

data_T1 <- do.call(cbind, lapply(1:length(timepoints), function(i)
{
  timepoint <- timepoints[i]
  id <- 2683120 + i
  dat <- read.table(paste0('GSM',id,'_T',timepoint,'_RTX_genes.fpkm_tracking'), header=T, sep='\t')
  return(dat$FPKM)
}
))
colnames(data_T1) <- timepoints
rownames(data_T1) <- read.table(paste0('GSM2683121_T00_RTX_genes.fpkm_tracking'), header=T, sep='\t')$gene_id

data_T2 <- do.call(cbind, lapply(1:length(timepoints), function(i)
{
  timepoint <- timepoints[i]
  id <- 2683132 + i
  dat <- read.table(paste0('GSM',id,'_RT',timepoint,'_RTX_genes.fpkm_tracking'), header=T, sep='\t')
  return(dat$FPKM)
}
))
colnames(data_T2) <- timepoints
rownames(data_T2) <- read.table(paste0('GSM2683133_RT00_RTX_genes.fpkm_tracking'), header=T, sep='\t')$gene_id

## Filter out completely 0 genes
zero_genes <- rowSums(data_U1) == 0 & rowSums(data_U2) == 0 & rowSums(data_T1) == 0 & rowSums(data_T2) == 0
data_U1 <- data_U1[!zero_genes,]
data_U2 <- data_U2[!zero_genes,]
data_T1 <- data_T1[!zero_genes,]
data_T2 <- data_T2[!zero_genes,]

dat <- data.frame(ID=rep('U1', nrow(data_U1)*length(timepoints)),
                  time=rep(timepoints, each=nrow(data_U1)),
                  gene=rep(rownames(data_U1),length(timepoints)),
                  expression=as.vector(data_U1))
dat <- rbind(dat, data.frame(ID=rep('U2', nrow(data_U2)*length(timepoints)),
                             time=rep(timepoints, each=nrow(data_U2)),
                             gene=rep(rownames(data_U2),length(timepoints)),
                             expression=as.vector(data_U2)))
dat <- rbind(dat, data.frame(ID=rep('T1', nrow(data_T1)*length(timepoints)),
                             time=rep(timepoints, each=nrow(data_T1)),
                             gene=rep(rownames(data_T1),length(timepoints)),
                             expression=as.vector(data_T1)))
dat <- rbind(dat, data.frame(ID=rep('T2', nrow(data_T2)*length(timepoints)),
                             time=rep(timepoints, each=nrow(data_T2)),
                             gene=rep(rownames(data_T2),length(timepoints)),
                             expression=as.vector(data_T2)))

write.csv(dat, file='preprocessed_Bcell_GSE100441.csv', row.names=F)
