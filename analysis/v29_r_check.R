# v29：核对表 S9 分层模型的系数个数（length(coef(fit))）
# 说明：v28 表 S9 分层行记 "7 + 0"，脚注却写「该协变量已从线性预测子中移除」，二者矛盾。
# 本脚本对两种写法各拟合一次并打印 length(coef())，以确定实际拟合对象。
# 运行：Rscript analysis/v29_r_check.R → analysis/v29_r_check_report.txt
suppressPackageStartupMessages(library(survival))
ex <- Sys.getenv("SLECI_EXPORT_DIR", "_cpexp3")
out <- file.path(Sys.getenv("SLECI_BASE", "."), "analysis", "v29_r_check_report.txt"
sink(out, split = TRUE)
cat("R", R.version.string, "| survival", as.character(packageVersion("survival")), "\n\n")

X  <- "expo + age + female + charlson_norheum + lupus_nephritis + renal_disease + diabetes"
Xa <- "expo + female + charlson_norheum + lupus_nephritis + renal_disease + diabetes"
Xc <- "expo + age + female + lupus_nephritis + renal_disease + diabetes"
rd <- function(n) read.csv(file.path(ex, paste0(n, ".csv")))
hr <- function(f) sprintf("k=%d n=%d ev=%d HR %.3f (%.3f-%.3f) p=%.4f",
                          length(coef(f)), summary(f)$n, summary(f)$nevent,
                          exp(coef(f)["expo"]), exp(confint(f)["expo", 1]),
                          exp(confint(f)["expo", 2]), summary(f)$coefficients["expo", "Pr(>|z|)"])

cat("========== 系数个数基线 ==========\n")
for (nm in c("f_main", "r_main")) {
  d <- rd(nm)
  f <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X)), data = d)
  cat(sprintf("%-8s 主模型               %s\n", nm, hr(f)))
}

cat("\n========== 分层：连续变量保留在模型内（v28 实际拟合对象） ==========\n")
for (nm in c("f_main", "r_main")) {
  d <- rd(nm)
  for (v in c("charlson_norheum", "age")) {
    br <- quantile(d[[v]], c(0, 1/3, 2/3, 1), na.rm = TRUE); br[1] <- br[1] - 0.5
    d$g <- cut(d[[v]], breaks = unique(br), include.lowest = TRUE)
    f <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X, "+ strata(g)")), data = d)
    cat(sprintf("%-8s strata(%s) 保留连续项  %s\n", nm, v, hr(f)))
  }
}

cat("\n========== 分层：连续变量从线性预测子移除（脚注所描述） ==========\n")
for (nm in c("f_main", "r_main")) {
  d <- rd(nm)
  for (v in c("charlson_norheum", "age")) {
    br <- quantile(d[[v]], c(0, 1/3, 2/3, 1), na.rm = TRUE); br[1] <- br[1] - 0.5
    d$g <- cut(d[[v]], breaks = unique(br), include.lowest = TRUE)
    xx <- if (v == "age") Xa else Xc
    f <- coxph(as.formula(paste("Surv(entry, stop, event) ~", xx, "+ strata(g)")), data = d)
    cat(sprintf("%-8s strata(%s) 移除连续项  %s\n", nm, v, hr(f)))
  }
}

cat("\n========== Schoenfeld 逐项 p（供表 S9 行内标注统一） ==========\n")
for (nm in c("f_main", "r_main")) {
  d <- rd(nm)
  f <- coxph(as.formula(paste("Surv(entry, stop, event) ~", X)), data = d)
  z <- cox.zph(f, transform = "km")
  cat("\n--", nm, "--\n"); print(round(z$table[, "p"], 4))
}
cat("\n[完成]\n")
sink()
