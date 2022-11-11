setwd("~/Documents/Python/imputation/GAIN")
library(missForest)
rm(list = ls())
data = read.csv('spam.csv', header = FALSE)
data.m = read.csv('spam_m.csv', header = FALSE, na=c(''))

data.imputed = missForest(data.m)$ximp
write.csv(data.imputed, 'spam_mf.csv', row.names = FALSE, col.names = FALSE)

normalization = apply(data, 2, range)
for (var in colnames(data)) {
  data[var] = (data[var] - normalization[1, var]) / (normalization[2, var] - normalization[1, var])
  data.imputed[var] = (data.imputed[var] - normalization[1, var]) / (normalization[2, var] - normalization[1, var])
}

print(sqrt(sum((data.imputed - data) ** 2) / sum(is.na(data.m))))