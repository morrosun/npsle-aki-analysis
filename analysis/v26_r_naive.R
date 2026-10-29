# 前置：python analysis/v25_export_cox.py → analysis/_cpexp2/*.csv
# 运行：Rscript analysis/v26_r_naive.R（核对 coxph 模型方差与聚类方差）
# 核对：coxph 不加 cluster 时的模型区间，与 PHReg（Python）是否一致
suppressPackageStartupMessages(library(survival))
EX <- Sys.getenv("SLECI_EXPORT_DIR", "_cpexp2")
rd <- function(nm) read.csv(file.path(EX, paste0(nm, ".csv")))
COV <- c("age","female","charlson_norheum","lupus_nephritis","renal_disease","diabetes")
fml <- as.formula(paste("Surv(entry, stop, event) ~ expo +", paste(COV, collapse=" + ")))
for (nm in c("f_main","r_main","f_noesrd","r_noesrd")) {
  d <- rd(nm)
  m1 <- coxph(fml, data=d, ties="efron")                 # 模型（naive）方差
  m2 <- coxph(fml, data=d, ties="efron", cluster=subject_id)  # 聚类稳健
  s1 <- summary(m1)$conf.int["expo",]
  s2 <- summary(m2)$conf.int["expo",]
  cat(sprintf("%-9s model %.4f (%.3f-%.3f) | cluster %.4f (%.3f-%.3f)\n",
              nm, s1[1], s1[3], s1[4], s2[1], s2[3], s2[4]))
}
