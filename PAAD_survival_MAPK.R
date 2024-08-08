# to run the script you must download TCGA pan-cancer PAAD dataset
library(survival)
library(survminer)
library(dplyr)
library(ggplot2)
library(stringr)

clin_raw <- read.delim("paad_tcga_pan_can_atlas_2018/data_clinical_patient.txt", sep = '\t',skip = 4)
rownames(clin_raw) <- clin_raw$PATIENT_ID

RNA_raw <- read.delim("paad_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt",check.names = FALSE)
RNA_raw[is.na(RNA_raw)] <- 0
RNA_raw <- RNA_raw[RNA_raw$Hugo_Symbol!='',]
RNA_raw <- RNA_raw[!duplicated(RNA_raw$Hugo_Symbol),]
rownames(RNA_raw) <- RNA_raw$Hugo_Symbol
RNA <- as.data.frame(t(RNA_raw[-1:-2]))
# Align clinical data:
clin <- clin_raw[str_sub(row.names(RNA), end = -4),]

# create a survival object consisting of times & censoring
surv_obj <- Surv(time = clin$OS_MONTHS, 
                 event = clin$OS_STATUS=="1:DECEASED")
fit <- survfit(surv_obj ~ 1, data = clin)
ggsurvplot(fit, data = clin, xlab = "Month", ylab = "Overall survival")


library(progeny)
zscores = as.matrix(t(RNA))
pathways <- progeny(zscores, scale=TRUE, organism="Human")  #, top = 100, perm = 1)
path_df = as.data.frame(pathways)

# now creating object without zero times
clin_filt <- clin[clin$OS_MONTHS > 0,]
RNA_filt <- RNA[clin$OS_MONTHS > 0,]
path_filt <- path_df[clin$OS_MONTHS > 0,]
# create a survival object consisting of times & censoring
surv_filt <- Surv(time = clin_filt$OS_MONTHS, 
                 event = clin_filt$OS_STATUS=="1:DECEASED")
fit <- survfit(surv_filt ~ 1, data = clin_filt)
ggsurvplot(fit, data = clin_filt, xlab = "Month", ylab = "Overall survival")


# plotting Kaplan-Mayer curves
pathway = 'MAPK'
pathway_data = path_filt$MAPK
# sort age 
uni_path = sort(unique(pathway_data))
# store results
results_path = matrix(1,length(uni_path))
# do a for loop for every unique value of age mat
for (i in 2:(length(uni_path)-1)){ # Starting from 2 because the first element would yield no elements higher than.
  path_i = 1*(pathway_data>uni_path[i])
  # survdiff is the function from the survival package 
  logrank = survdiff(surv_filt ~ path_i)
  # store in results_age
  results_path[i] = logrank$pvalue
}
# Plot unique elements of age against p-value
plot(uni_path, results_path, log = 'y')
# Select minimum P-value
min_p_path = which.min(results_path)
# here are 2 good thresholds, -1 and 1
opt_thr = uni_path[min_p_path]
#opt_JAK = opt_thr
#opt_thr = 0.0
# I recalculated the P-value as a sanity check
pval = survdiff(surv_filt ~ pathway_data>opt_thr)$pvalue
nplus = sum(pathway_data>opt_thr)   # how many patients we have in high group
nminus = sum(pathway_data<opt_thr)   # how many patients we have in low group
# fit Kaplan Meier model
path_filt <- path_filt %>% mutate(MAPK_group = ifelse(MAPK >= 1, "high", ifelse(MAPK < 1.001*opt_thr,"low","intermediate")))
nhigh = sum(pathway_data>1)
ninter = sum((pathway_data<1) & (pathway_data > 1.001*opt_thr))
nlow = sum(pathway_data<1.001*opt_thr)
#KM = survfit(surv_filt ~ pathway_data>opt_thr,data = path_filt)
KM = survfit(surv_filt ~ MAPK_group,data = path_filt)
# Plot Kaplan Meier 
#plot(KM, lwd = 3, col = c(1,2), cex.axis = 1.5, xlab = 'Months', ylab = 'Survival Probability' , cex.lab = 1.5)
#ggsurvplot(KM, data = path_filt,pval = TRUE,xlab = 'Overall survival time, months',
#           legend.labs=c(paste('Low ',pathway,' activity, ',nminus,' patient(s)',sep = ''),paste('High ',pathway,' activity, ',nplus,' patient(s)',sep = '')),
#           palette = c('blue','red'),legend.title="")
p <- ggsurvplot(KM, data = path_filt,pval = TRUE,xlab = 'Overall survival time, months',
           legend.labs=c(paste("High MAPK activity,\n",nhigh," patients",sep=""),paste("Intermediate MAPK activity,\n",ninter," patients",sep=""),paste("Low MAPK activity,\n",nlow," patients",sep="")),
           legend.title=""
           )

ggpar(p, 
      font.main = c(13, "bold"),
      font.x = c(16, "bold"),
      font.y = c(16, "bold"),
      font.caption = c(16, "bold"), 
      font.legend = c(13, "bold"), 
      font.tickslab = c(14, "bold"))

