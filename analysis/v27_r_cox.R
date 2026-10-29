# v27 PH 复核：正确的 tt() 时变系数实现
# 前置：python analysis/v27_export_cox.py → analysis/_cpexp3/*.csv
# 运行：Rscript analysis/v27_r_cox.R → v27_r_cox_report.txt
#
# 为什么重写：v25_r_cox.R 用 coxph(..., tt = list(age = f, ...)) 的方式传 tt，
# 但 survival 包的 tt= 参数是**一个**函数，只有在公式里写 tt(x) 时才被调用。
# 实测 nterm 仍为 7、loglik 完全相同（LR p=1），说明该模型并未拟合任何时变项，
# 表 S9 的 "tt" 行只是原模型的重述。本脚本按 survival 文档的正确写法实现。
suppressPackageStartupMessages(library(survival))
ex <- Sys.getenv("SLECI_EXPORT_DIR", "_cpexp3")
out <- file.path(Sys.getenv("SLECI_BASE", "."), "analysis", "v27_r_cox_report.txt"
sink(out, split = TRUE)
cat("R", R.version.string, "| survival", as.character(packageVersion("survival")), "\n\n")

X <- "expo + age + female + charlson_norheum + lupus_nephritis + renal_disease + diabetes"
rd <- function(n) read.csv(file.path(ex, paste0(n, ".csv")))

show <- function(fit, lab, clust = TRUE) {
  s <- summary(fit)
  cat(sprintf("%-16s n %3d events %3d | coef %.8f HR %.3f (%.3f-%.3f) p=%.4f\n",
              lab, s$n, s$nevent,
              coef(fit)["expo"], exp(coef(fit)["expo"]),
              exp(confint(fit)["expo", 1]), exp(confint(fit)["expo", 2]),
              s$coefficients["expo", "Pr(>|z|)"]))
  invisible(fit)
}

cat("========== 1. 主模型（严格 CAM 窗口，n=326/326） ==========\n")
for (nm in c("f_main", "r_main", "f_noesrd", "r_noesrd")) {
  d <- rd(nm)
  fit <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X)), data = d)
  fitc <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X)), data = d, cluster = subject_id)
  cat(sprintf("%-9s rows %4d stays %3d events %3d | coef %.10f loglik %.6f\n",
              nm, nrow(d), length(unique(d$stay_id)), sum(d$event), coef(fit)["expo"], fit$loglik[2]))
  show(fit, paste0(nm, " [model]"))
  show(fitc, paste0(nm, " [cluster]"))
}

cat("\n========== 2. Schoenfeld 残差（KM 变换） ==========\n")
for (nm in c("f_main", "r_main")) {
  d <- rd(nm)
  z <- cox.zph(coxph(as.formula(paste("Surv(entry, stop, event) ~", X)), data = d), transform = "km")
  cat("\n--", nm, "--\n"); print(round(z$table, 4))
}

cat("\n========== 3. 时变系数（正确的 tt() 写法） ==========\n")
# tt= 是一个函数；公式里必须出现 tt(var)。t 为该风险集的事件时间（小时，自入 ICU 起）。
TT <- function(x, t, ...) x * log1p(t)
XTT <- paste(X, "+ tt(age) + tt(charlson_norheum) + tt(lupus_nephritis)")
for (nm in c("f_main", "r_main")) {
  d <- rd(nm)
  fit <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X)), data = d)
  f2 <- coxph(as.formula(paste("Surv(entry, stop, event) ~", XTT)), data = d, tt = TT)
  cat(sprintf("\n-- %s --\n  base: nterm %d loglik %.6f\n  tt  : nterm %d loglik %.6f\n",
              nm, length(coef(fit)), fit$loglik[2], length(coef(f2)), f2$loglik[2]))
  cat("  tt 项:", paste(setdiff(names(coef(f2)), names(coef(fit))), collapse = ", "), "\n")
  cat(sprintf("  LR 检验 base vs tt: chisq %.4f df %d p=%.6g\n",
              2 * (f2$loglik[2] - fit$loglik[2]),
              length(coef(f2)) - length(coef(fit)),
              pchisq(2 * (f2$loglik[2] - fit$loglik[2]),
                     length(coef(f2)) - length(coef(fit)), lower.tail = FALSE)))
  show(f2, paste0(nm, " [tt 3 项]"))
  # 仅对 zph p<0.05 的项加 tt
  s <- summary(f2)$coefficients
  cat("  时变项系数（log-HR per log(h+1)）:\n")
  for (k in grep("^tt\\(", rownames(s), value = TRUE))
    cat(sprintf("    %-28s %.4f (%.4f-%.4f) p=%.4f\n", k, s[k, "coef"],
                s[k, "coef"] - 1.96 * s[k, "se(coef)"], s[k, "coef"] + 1.96 * s[k, "se(coef)"],
                s[k, "Pr(>|z|)"]))
}

cat("\n========== 4. 分层（Charlson / 年龄三分位） ==========\n")
for (nm in c("f_main", "r_main")) {
  d <- rd(nm); cat("\n--", nm, "--\n")
  for (v in c("charlson_norheum", "age")) {
    br <- quantile(d[[v]], c(0, 1/3, 2/3, 1), na.rm = TRUE); br[1] <- br[1] - 0.5
    d$g <- cut(d[[v]], breaks = unique(br), include.lowest = TRUE)
    f <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X, "+ strata(g)")), data = d)
    cat(sprintf("  按 %-16s 分层（%d 层）: expo HR %.3f (%.3f-%.3f) p=%.4f\n", v,
                length(unique(d$g)), exp(coef(f)["expo"]), exp(confint(f)["expo", 1]),
                exp(confint(f)["expo", 2]), summary(f)$coefficients["expo", "Pr(>|z|)"]))
  }
}

cat("\n========== 5. 全精度（供与 Python PHReg 对照） ==========\n")
for (nm in c("f_main", "r_main")) {
  d <- rd(nm)
  fit <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X)), data = d)
  cat(sprintf("%s: n=%d events=%d coef=%.10f HR=%.6f loglik=%.6f ties=%s\n",
              nm, fit$n, fit$nevent, coef(fit)["expo"], exp(coef(fit)["expo"]),
              fit$loglik[2], fit$method))
}
cat("\n[完成]\n")
sink()
