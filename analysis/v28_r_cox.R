# v28 PH 复核：明确区分「仅对被 flag 的协变量加 tt()」与「对三项都加 tt()」
# 前置：Rscript analysis/v27_export_cox.py → analysis/_cpexp3/*.csv（与 v27 相同数据）
# 运行：Rscript analysis/v28_r_cox.R → analysis/v28_r_cox_report.txt
#
# 上一版（v27）表 S9 的前向行标为「仅 lupus nephritis」，但实际拟合对象是三项 tt()，
# 文字与表格不符。本脚本把两种规格分别拟合、分别报告，避免 HR 与 LR 检验错配。
suppressPackageStartupMessages(library(survival))
ex <- Sys.getenv("SLECI_EXPORT_DIR", "_cpexp3")
out <- file.path(Sys.getenv("SLECI_BASE", "."), "analysis", "v28_r_cox_report.txt"
sink(out, split = TRUE)
cat("R", R.version.string, "| survival", as.character(packageVersion("survival")), "\n\n")

X <- "expo + age + female + charlson_norheum + lupus_nephritis + renal_disease + diabetes"
rd <- function(n) read.csv(file.path(ex, paste0(n, ".csv")))
TT <- function(x, t, ...) x * log1p(t)

row <- function(fit, lab, df_extra = 0, base = NULL) {
  s <- summary(fit)
  ci <- exp(confint(fit)["expo", ])
  txt <- sprintf("%-52s k=%2d n=%3d ev=%3d | HR %.3f (%.3f-%.3f) p=%.4f",
                 lab, length(coef(fit)), s$n, s$nevent,
                 exp(coef(fit)["expo"]), ci[1], ci[2],
                 s$coefficients["expo", "Pr(>|z|)"])
  if (!is.null(base)) {
    lr <- 2 * (fit$loglik[2] - base$loglik[2])
    txt <- paste0(txt, sprintf(" | LR chisq %.3f df %d p=%.4f", lr, df_extra,
                               pchisq(lr, df_extra, lower.tail = FALSE)))
  }
  cat(txt, "\n")
}

cat("========== 1. 主模型与 Schoenfeld（严格 CAM 窗口） ==========\n")
fits <- list(); zph <- list()
for (nm in c("f_main", "r_main")) {
  d <- rd(nm)
  fits[[nm]] <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X)), data = d)
  zph[[nm]] <- cox.zph(fits[[nm]], transform = "km")
  row(fits[[nm]], paste0(nm, " 主模型"))
  fitc <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X)), data = d, cluster = subject_id)
  ci <- exp(confint(fitc)["expo", ])
  cat(sprintf("%-52s k=%2d n=%3d ev=%3d | HR %.3f (%.3f-%.3f) p=%.4f  [cluster]\n",
              paste0(nm, " 主模型"), length(coef(fitc)), summary(fitc)$n, summary(fitc)$nevent,
              exp(coef(fitc)["expo"]), ci[1], ci[2],
              summary(fitc)$coefficients["expo", "Pr(>|z|)"]))
  cat("\n-- cox.zph (", nm, ") --\n", sep = ""); print(round(zph[[nm]]$table, 4))
  cat("\n")
}

cat("========== 2. 前向：只对 lupus nephritis 加 tt()（k=8，LR df=1） ==========\n")
d <- rd("f_main")
f1 <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X, "+ tt(lupus_nephritis)")),
            data = d, tt = TT)
row(f1, "forward: tt(lupus_nephritis) only", 1, fits$f_main)
cat(sprintf("    tt(lupus_nephritis) 系数 %.4f (SE %.4f) p=%.4f\n",
            coef(f1)["tt(lupus_nephritis)"], sqrt(vcov(f1)["tt(lupus_nephritis)", "tt(lupus_nephritis)"]),
            summary(f1)$coefficients["tt(lupus_nephritis)", "Pr(>|z|)"]))

cat("\n========== 3. 前向：三项都加 tt()（k=10，LR df=3） ==========\n")
f3 <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X,
                             "+ tt(age) + tt(charlson_norheum) + tt(lupus_nephritis)")),
            data = d, tt = TT)
row(f3, "forward: tt(age)+tt(charlson)+tt(lupus)", 3, fits$f_main)
for (k in grep("^tt\\(", names(coef(f3)), value = TRUE))
  cat(sprintf("    %-26s %.4f (SE %.4f) p=%.4f\n", k, coef(f3)[k],
              sqrt(vcov(f3)[k, k]), summary(f3)$coefficients[k, "Pr(>|z|)"]))

cat("\n========== 4. 反向：三项都加 tt()（k=10，LR df=3） ==========\n")
d <- rd("r_main")
g3 <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X,
                             "+ tt(age) + tt(charlson_norheum) + tt(lupus_nephritis)")),
            data = d, tt = TT)
row(g3, "reverse: tt(age)+tt(charlson)+tt(lupus)", 3, fits$r_main)
for (k in grep("^tt\\(", names(coef(g3)), value = TRUE))
  cat(sprintf("    %-26s %.4f (SE %.4f) p=%.4f\n", k, coef(g3)[k],
              sqrt(vcov(g3)[k, k]), summary(g3)$coefficients[k, "Pr(>|z|)"]))

cat("\n========== 5. 反向：只对 Charlson 加 tt()（k=8，LR df=1，供核对） ==========\n")
g1 <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X, "+ tt(charlson_norheum)")),
            data = d, tt = TT)
row(g1, "reverse: tt(charlson_norheum) only", 1, fits$r_main)

cat("\n========== 6. 分层（Charlson / 年龄三分位） ==========\n")
for (nm in c("f_main", "r_main")) {
  d <- rd(nm); cat("\n--", nm, "--\n")
  for (v in c("charlson_norheum", "age")) {
    br <- quantile(d[[v]], c(0, 1/3, 2/3, 1), na.rm = TRUE); br[1] <- br[1] - 0.5
    d$g <- cut(d[[v]], breaks = unique(br), include.lowest = TRUE)
    f <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X, "+ strata(g)")), data = d)
    ci <- exp(confint(f)["expo", ])
    cat(sprintf("  %-18s %d 层: HR %.3f (%.3f-%.3f) p=%.4f\n", v, length(unique(d$g)),
                exp(coef(f)["expo"]), ci[1], ci[2],
                summary(f)$coefficients["expo", "Pr(>|z|)"]))
  }
}
cat("\n[完成]\n")
sink()
