# v24 时序分析：R survival 独立复算（表 4 Panel B 的数据来源）
# 前置：先运行 v24_export_cox.py 生成 counting-process 导出集
#   python analysis/v24_export_cox.py   →  analysis/_cpexp/*.csv + meta.json
# 运行：Rscript analysis/v24_r_cox_full.R  →  v24_r_cox_full_report.txt
suppressPackageStartupMessages(library(survival))
ex <- Sys.getenv("SLECI_EXPORT_DIR", "_cpexp")
out <- file.path(Sys.getenv("SLECI_BASE", "."), "analysis", "v24_r_cox_full_report.txt")
sink(out, split = TRUE)
cat("R", R.version.string, "| survival", as.character(packageVersion("survival")), "\n\n")

covs <- c("age", "female", "charlson_norheum", "lupus_nephritis", "renal_disease", "diabetes")
files <- c("f_main", "f_tv2", "f_tv3", "f_aki2", "f_72", "f_24h", "f_uta", "f_lm24",
           "f_period", "f_first",
           "r_main", "r_tv2", "r_tv3", "r_72", "r_24h", "r_nass", "r_ge4",
           "r_period", "r_first")

lab <- c(f_main = "F 主模型", f_tv2 = "F + 升压药/通气（时变）", f_tv3 = "F + sepsis（时变）",
         f_aki2 = "F 结局 KDIGO>=2", f_72 = "F 前 72 h", f_24h = "F 首次评估 <=24 h",
         f_uta = "F 暴露含 UTA", f_lm24 = "F 24 h landmark", f_period = "F 48 h 分段",
         f_first = "F 仅首次合格入住",
         r_main = "R 主模型", r_tv2 = "R + 升压药/通气（时变）", r_tv3 = "R + sepsis（时变）",
         r_72 = "R 前 72 h", r_24h = "R 首次评估 <=24 h", r_nass = "R + 评估次数",
         r_ge4 = "R 评估 >=4 次", r_period = "R 48 h 分段", r_first = "R 仅首次合格入住")

fmt <- function(m, term) {
  s <- summary(m)
  sprintf("%.2f (%.2f-%.2f) p=%.2f", s$conf.int[term, "exp(coef)"],
          s$conf.int[term, "lower .95"], s$conf.int[term, "upper .95"],
          s$coefficients[term, "Pr(>|z|)"])
}

for (f in files) {
  d <- read.csv(file.path(ex, paste0(f, ".csv")))
  d$event <- as.integer(d$event)
  extra <- intersect(c("vaso_tv", "vent_tv", "sep_tv", "n_assess"), names(d))
  rhs <- paste(c("expo", extra, covs), collapse = " + ")
  fml <- as.formula(paste("Surv(entry, stop, event) ~", rhs))
  m <- coxph(fml, data = d, ties = "efron")
  mc <- coxph(fml, data = d, ties = "efron", cluster = subject_id)
  nstay <- length(unique(d$stay_id)); nsub <- length(unique(d$subject_id))

  if (f %in% c("f_period", "r_period")) {
    d$expo_after48 <- d$expo * as.integer(d$period == 2)
    fml2 <- as.formula(paste("Surv(entry, stop, event) ~ expo + expo_after48 +",
                             paste(c(covs), collapse = " + ")))
    mp <- coxph(fml2, data = d, ties = "efron")
    sp <- summary(mp); b <- coef(mp); V <- vcov(mp)
    late <- b["expo"] + b["expo_after48"]
    se_late <- sqrt(V["expo", "expo"] + V["expo_after48", "expo_after48"] +
                      2 * V["expo", "expo_after48"])
    s1 <- sp$conf.int["expo", ]; s2 <- sp$conf.int["expo_after48", ]
    pp <- sp$coefficients["expo_after48", "Pr(>|z|)"]
    line <- sprintf("%-24s stays %3d subj %3d events %3d | 0-48h %.2f (%.2f-%.2f) | >48h %.2f (%.2f-%.2f) | ratio %.2f (%.2f-%.2f) p=%.3f",
                    lab[[f]], nstay, nsub, sum(d$event),
                    s1["exp(coef)"], s1["lower .95"], s1["upper .95"],
                    exp(late), exp(late - 1.96 * se_late), exp(late + 1.96 * se_late),
                    s2["exp(coef)"], s2["lower .95"], s2["upper .95"], pp)
    p0 <- sp$coefficients["expo", "Pr(>|z|)"]
    cat(line, sprintf("| p(0-48h)=%.3f", p0), "\n")
    next
  }

  cat(sprintf("%-24s stays %3d subj %3d events %3d | %s | cluster %s\n",
              lab[[f]], nstay, nsub, sum(d$event), fmt(m, "expo"), fmt(mc, "expo")))
}

cat("\n---- Schoenfeld 残差检验（主模型）\n")
for (f in c("f_main", "r_main")) {
  d <- read.csv(file.path(ex, paste0(f, ".csv")))
  d$event <- as.integer(d$event)
  m <- coxph(as.formula(paste("Surv(entry, stop, event) ~",
                              paste(c("expo", covs), collapse = " + "))),
             data = d, ties = "efron")
  z <- cox.zph(m, transform = "km", terms = FALSE)
  cat(sprintf("  %s: chisq=%.2f df=%d p=%.3f\n", f,
              z$table["expo", "chisq"], z$table["expo", "df"], z$table["expo", "p"]))
  cat(sprintf("  %s 全局检验 p=%.3f\n", f, z$table["GLOBAL", "p"]))
}
sink()
cat("done\n")
