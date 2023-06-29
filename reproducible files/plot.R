library(ggplot2)
library(gtable)
library(cowplot)
library(grid)
rm(list = ls())
gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

result = read.csv('../result_bn.csv')
result.summary = data.frame(datasize = integer(), missingtype = character(), `Missing rate` = numeric(), algorithm = character(), rmse = numeric(), F1 = numeric(), check.names = FALSE)
for (datasize in c(500, 1000, 2000, 3000)) {
  for (missingrate in c(0.1, 0.3, 0.5)) {
    result.summary[nrow(result.summary) + 1, ] = list(datasize, 'MCAR', missingrate, 'Complete', 0, mean(result[(result$datasize == datasize) & (result$missingtype == 'Complete'), 'F1']), 0, 0)
    result.summary[nrow(result.summary) + 1, ] = list(datasize, 'MAR', missingrate, 'Complete', 0, mean(result[(result$datasize == datasize) & (result$missingtype == 'Complete'), 'F1']), 0, 0)
    result.summary[nrow(result.summary) + 1, ] = list(datasize, 'MNAR', missingrate, 'Complete', 0, mean(result[(result$datasize == datasize) & (result$missingtype == 'Complete'), 'F1']), 0, 0)
    for (missingtype in c('Mean', 'KNN', 'GAIN', 'SoftImpute', 'MF', 'MBMF')) {
      for (algorithm in algorithm.list) {
        result.summary[nrow(result.summary) + 1, ] = list(datasize, missingtype, missingrate, algorithm, mean(result[(result$datasize == datasize) & (result$missingtype == missingtype) & (result$missingrate == missingrate) & (result$algorithm == algorithm), 'rmse']), mean(result[(result$datasize == datasize) & (result$missingtype == missingtype) & (result$missingrate == missingrate) & (result$algorithm == algorithm), 'F1']))
      }
    }
  }
}
result.summary$missingtype = factor(result.summary$missingtype, levels = c('MCAR', 'MAR', 'MNAR'))
result.summary$algorithm = factor(result.summary$algorithm, levels = c('MF', 'MBMF', 'Mean', 'KNN', 'GAIN', 'softImpute', 'Complete'))
result.summary[result.summary$`Missing rate` == 0.1, 'Missing rate'] = 'Missing rate 10%'
result.summary[result.summary$`Missing rate` == 0.3, 'Missing rate'] = 'Missing rate 30%'
result.summary[result.summary$`Missing rate` == 0.5, 'Missing rate'] = 'Missing rate 50%'


p1 = ggplot(result.summary[result.summary$algorithm != 'Complete', ], aes(x=datasize, y=rmse, group=algorithm, color=algorithm)) + geom_line(size=0.7) + geom_point(size=2, aes(shape=algorithm)) + scale_color_manual(values = gg_color_hue(7)[1:6]) + facet_grid(`Missing rate`~missingtype, labeller = label_wrap_gen(width = 15)) + ylab('RMSE') + xlab('Data size') + theme_bw() + theme(panel.grid.minor = element_blank(), text = element_text(size = 9), legend.position = 'right', legend.title = element_blank(), strip.background = element_blank(), strip.text = element_text(size = 8))
pdf('../bn_rmse.pdf', width = 6.5, height = 5)
grid.draw(p1)
dev.off()

p2 = ggplot(data = result.summary, aes(x = datasize, y = F1, group = algorithm, color=algorithm)) + geom_line(size = 0.7, aes(linetype=algorithm)) + geom_point(size=2, aes(shape=algorithm)) + scale_linetype_manual(values=c("solid", "solid", "solid", "solid", "solid", "solid", 'dotted')) + geom_point() + facet_grid(`Missing rate`~missingtype, labeller = label_wrap_gen(width = 15)) + ylab('F1') + xlab('Data size') + theme_bw() + theme(panel.grid.minor = element_blank(), text = element_text(size = 9), legend.position = 'right', legend.title = element_blank(), strip.background = element_blank(), strip.text = element_text(size = 8))
pdf('../bn_F1.pdf', width = 6.5, height = 5)
grid.draw(p2)
dev.off()
