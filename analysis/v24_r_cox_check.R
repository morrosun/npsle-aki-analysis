suppressPackageStartupMessages(library(survival))
ana <- file.path(Sys.getenv("SLECI_BASE", "."), "analysis")
out <- file.path(ana, "v24_r_cox_report.txt")
sink(out, split = TRUE)

cat("R", R.version.string, "| survival", as.character(packageVersion("survival")), "\n")
cat(strrep("=", 78), "\n")
cat("独立交叉验证：Python statsmodels PHReg（Efron）+ 自写三明治 vs R coxph\n")
cat(strrep("=", 78), "\n\n")

covs <- c("age", "female", "charlson_norheum", "lupus_nephritis", "renal_disease", "diabetes")

for (nm in c("forward", "reverse")) {
  d <- read.csv(file.path(ana, paste0("_cp_", nm, ".csv")))
  d$event <- as.integer(d$event)
  f <- as.formula(paste("Surv(entry, stop, event) ~ expo +", paste(covs, collapse = " + ")))
  cat("---- ", nm, " (rows ", nrow(d), ", events ", sum(d$event),
      ", patients ", length(unique(d$subject_id)), ")\n", sep = "")

  m0 <- coxph(f, data = d, ties = "efron")
  s0 <- summary(m0)
  cat(sprintf("  coxph, Efron, model SE      : HR %.2f (%.2f-%.2f) p=%.3f\n",
              s0$conf.int["expo", "exp(coef)"], s0$conf.int["expo", "lower .95"],
              s0$conf.int["expo", "upper .95"], s0$coefficients["expo", "Pr(>|z|)"]))

  m1 <- coxph(f, data = d, ties = "efron", cluster = subject_id)
  s1 <- summary(m1)
  cat(sprintf("  coxph, Efron, cluster(id)   : HR %.2f (%.2f-%.2f) p=%.3f  [n clusters %d]\n",
              s1$conf.int["expo", "exp(coef)"], s1$conf.int["expo", "lower .95"],
              s1$conf.int["expo", "upper .95"], s1$coefficients["expo", "Pr(>|z|)"],
              length(unique(d$subject_id))))

  m2 <- coxph(f, data = d, ties = "breslow", cluster = subject_id)
  s2 <- summary(m2)
  cat(sprintf("  coxph, Breslow, cluster(id) : HR %.2f (%.2f-%.2f) p=%.3f\n",
              s2$conf.int["expo", "exp(coef)"], s2$conf.int["expo", "lower .95"],
              s2$conf.int["expo", "upper .95"], s2$coefficients["expo", "Pr(>|z|)"]))

  # 分段：两个分段时间变暴露，无 period 主效应
  dp <- read.csv(file.path(ana, paste0("_cp_", nm, "_period.csv")))
  dp$event <- as.integer(dp$event)
  # 交互参数化：expo_any + expo_any×I(period 2)，第二项系数即 log(HR_late/HR_early)
  dp$expo_after48 <- dp$expo * as.integer(dp$period == 2)
  fp <- as.formula(paste("Surv(entry, stop, event) ~ expo + expo_after48 +",
                         paste(covs, collapse = " + ")))
  mp <- coxph(fp, data = dp, ties = "efron")
  sp <- summary(mp)
  cat(sprintf("  period 0-48h  : HR %.2f (%.2f-%.2f)\n",
              sp$conf.int["expo", "exp(coef)"], sp$conf.int["expo", "lower .95"],
              sp$conf.int["expo", "upper .95"]))
  hr_late <- exp(coef(mp)["expo"] + coef(mp)["expo_after48"])
  V <- vcov(mp); b <- coef(mp)
  se_late <- sqrt(V["expo", "expo"] + V["expo_after48", "expo_after48"] +
                    2 * V["expo", "expo_after48"])
  cat(sprintf("  period >48h   : HR %.2f (%.2f-%.2f)\n", hr_late,
              exp(log(hr_late) - 1.96 * se_late), exp(log(hr_late) + 1.96 * se_late)))
  cat(sprintf("  ratio >48h / 0-48h : %.2f (%.2f-%.2f), p=%.3f\n",
              sp$conf.int["expo_after48", "exp(coef)"],
              sp$conf.int["expo_after48", "lower .95"],
              sp$conf.int["expo_after48", "upper .95"],
              sp$coefficients["expo_after48", "Pr(>|z|)"]))
  # Schoenfeld 残差 PH 检验（R 能直接做）
  zph <- cox.zph(m0, transform = "km", terms = FALSE)
  cat(sprintf("  cox.zph (Schoenfeld) for expo: chisq=%.2f, df=%d, p=%.3f\n",
              zph$table["expo", "chisq"], zph$table["expo", "df"],
              zph$table["expo", "p"]))
  cat("\n")
}

sink()
cat("\nwritten:", out, "\n")
