# v25 审计：ESRD 排除、PH 专项复核、Python/R 未四舍五入对照
# 前置：python analysis/v25_export_cox.py  →  analysis/_cpexp2/*.csv
# 前置：python analysis/v25_export_cox.py → analysis/_cpexp2/*.csv（含 esrd 标志）
# 运行：Rscript analysis/v25_r_cox.R → v25_r_cox_report.txt
suppressPackageStartupMessages(library(survival))
ex <- Sys.getenv("SLECI_EXPORT_DIR", "_cpexp2")
out <- file.path(Sys.getenv("SLECI_BASE", "."), "analysis", "v25_r_cox_report.txt")
sink(out, split = TRUE)
cat("R", R.version.string, "| survival", as.character(packageVersion("survival")), "\n\n")

covs <- c("age", "female", "charlson_norheum", "lupus_nephritis", "renal_disease", "diabetes")
X <- "expo + age + female + charlson_norheum + lupus_nephritis + renal_disease + diabetes"

rd <- function(n) {
  d <- read.csv(file.path(ex, paste0(n, ".csv")))
  d
}

show <- function(fit, d, lab) {
  s <- summary(fit)
  cat(sprintf("%-14s n %3d events %3d | expo %.4f SE %.4f HR %.3f (%.3f-%.3f) p=%.4f\n",
              lab, s$n, s$nevent, coef(fit)["expo"], sqrt(vcov(fit)["expo", "expo"]),
              exp(coef(fit)["expo"]), exp(confint(fit)["expo", 1]), exp(confint(fit)["expo", 2]),
              s$coefficients["expo", "Pr(>|z|)"]))
  invisible(fit)
}

cat("========== 1. 主模型与 ESRD 排除 ==========\n")
for (nm in c("f_main", "r_main", "f_noesrd", "r_noesrd")) {
  d <- rd(nm)
  fit <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X)), data = d)
  fitc <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X)), data = d, cluster = subject_id)
  cat(sprintf("%-9s rows %4d stays %3d | coef %.8f loglik %.4f logtest p %.4g\n",
              nm, nrow(d), length(unique(d$stay_id)), coef(fit)["expo"],
              fit$loglik[2], summary(fit)$logtest["pvalue"]))
  show(fitc, d, nm)
  cat(sprintf("   共线/收敛: %s | iterations %d\n",
              ifelse(is.null(fit$warning), "ok", fit$warning), fit$iter[1]))
}

cat("\n========== 2. Schoenfeld 逐协变量（主模型） ==========\n")
for (nm in c("f_main", "r_main")) {
  d <- rd(nm)
  fit <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X)), data = d)
  z <- cox.zph(fit, transform = "km")
  cat("\n--", nm, "--\n")
  print(round(z$table, 4))
}

cat("\n========== 3. PH 专项：对异常协变量加时变系数 / 分层后暴露 HR 的变化 ==========\n")
# 3a：对 charlson 与 age 加 tt() 时变系数（log 时间尺度）
for (nm in c("f_main", "r_main")) {
  d <- rd(nm)
  fit <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X)), data = d)
  z <- cox.zph(fit, transform = "km")
  cat("\n--", nm, "--\n")
  print(round(z$table, 4))
}

cat("\n========== 3. PH 专项：对异常协变量加时变系数 / 分层后暴露 HR 的变化 ==========\n")
# 3a：对 charlson 与 age 加 tt() 时变系数（log 时间尺度）
ttfit <- function(d, tvars) {
  tt_spec <- lapply(tvars, function(v) {
    f <- eval(parse(text = sprintf("function(x, t, ...) x * log(t + 1)")))
    f
  })
  names(tt_spec) <- tvars
  ttarg <- as.call(c(list(quote(coxph)), list(formula = as.formula(paste("Surv(entry, stop, event) ~", X)),
                                             data = quote(d), tt = as.call(c(list(quote(list)), tt_spec)))))
  eval(ttarg)
}
for (nm in c("f_main", "r_main")) {
  d <- rd(nm)
  fit <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X)), data = d)
  z <- cox.zph(fit, transform = "km")$table
  pv <- z[rownames(z) != "GLOBAL", "p"]
  flag <- names(pv)[pv < 0.05]
  cat("\n--", nm, "：单协变量 p<0.05 的项 =",
      ifelse(length(flag) == 0, "无", paste(flag, collapse = ", ")), "\n")
  if (length(flag) > 0) {
    tt_spec <- lapply(flag, function(v) function(x, t, ...) x * log(t + 1))
    names(tt_spec) <- flag
    f2 <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X)), data = d, tt = tt_spec)
    cat(sprintf("   + tt(log t) 于 {%s}: expo HR %.3f (%.3f-%.3f) p=%.4f\n",
                paste(flag, collapse = ","),
                exp(coef(f2)["expo"]), exp(confint(f2)["expo", 1]), exp(confint(f2)["expo", 2]),
                summary(f2)$coefficients["expo", "Pr(>|z|)"]))
  }
  # 3b：按 Charlson 三分位分层
  br <- quantile(d$charlson_norheum, c(0, 1/3, 2/3, 1), na.rm = TRUE)
  br[1] <- br[1] - 0.5
  d$chg <- cut(d$charlson_norheum, breaks = unique(br), include.lowest = TRUE)
  f3 <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X, "+ strata(chg)")), data = d)
  cat(sprintf("   按 Charlson 三分位分层（%d 层）: expo HR %.3f (%.3f-%.3f) p=%.4f\n",
              length(unique(d$chg)),
              exp(coef(f3)["expo"]), exp(confint(f3)["expo", 1]), exp(confint(f3)["expo", 2]),
              summary(f3)$coefficients["expo", "Pr(>|z|)"]))
  # 3c：按年龄三分位分层
  br2 <- quantile(d$age, c(0, 1/3, 2/3, 1), na.rm = TRUE)
  br2[1] <- br2[1] - 0.5
  d$agh <- cut(d$age, breaks = unique(br2), include.lowest = TRUE)
  f4 <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X, "+ strata(agh)")), data = d)
  cat(sprintf("   按年龄三分位分层（%d 层）: expo HR %.3f (%.3f-%.3f) p=%.4f\n",
              length(unique(d$agh)),
              exp(coef(f4)["expo"]), exp(confint(f4)["expo", 1]), exp(confint(f4)["expo", 2]),
              summary(f4)$coefficients["expo", "Pr(>|z|)"]))
}

cat("\n========== 4. 未四舍五入对照（供与 Python PHReg 比对） ==========\n")
for (nm in c("f_main", "r_main")) {
  d <- rd(nm)
  fit <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X)), data = d)
  cat(sprintf("%s: n=%d events=%d coef_expo=%.10f loglik=%.6f iters=%d\n",
              nm, fit$n, fit$nevent, coef(fit)["expo"], fit$loglik[2], fit$iter[1]))
  cat(sprintf("   ties=%s, concat p值尺度 logtest p=%.6g\n", fit$method,
              summary(fit)$logtest["pvalue"]))
}
cat("\n[完成]\n")
sink()
