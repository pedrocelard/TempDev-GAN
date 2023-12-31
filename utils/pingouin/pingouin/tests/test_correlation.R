library(correlation)

x <- c(4.524991087851508, 4.420531811767872, 3.778970926584845, 12.0, 3.2553302798031223, 4.649569372128418, 2.951786225128683, 4.591357332700219, 1.0489956831264005, 2.9292390843995104, 2.6738018656932527, 4.31118852334745, 5.406716105951538, 3.827585880548361, 4.510669777362404, 5.48020001172895, 5.897501992923205, 3.2481046204119224, 3.6896719823018116, 4.659839225398035, 5.492978998780082, 4.093017621645702, 3.702447504183336, 1.6755435235042087, 2.1236637738250335, 5.622025310958057, 2.7972808778327707, 3.495237872288203, 2.418519025293012, 2.184008274957259)
y <- c(7.417043974095821, 5.073260861980091, 7.2560606695752785, 7.978672340219759, 4.480094096479011, -8.0, 4.380334906909793, 6.202861741341518, 5.004916622004917, 5.274654700615849, 6.007153126165779, 7.362881992968021, 6.836293821056158, 4.549735015009455, 5.7398927665282, 4.977065819368747, 7.271512763937, 5.092800144396517, 6.305237088575375, 6.913523215921814, 5.947704426177541, 6.606245187335942, 5.691865988238505, 4.044863387178897, 6.125520033132931, 6.6929048900606105, 4.083471867309398, 6.451663151111775, 5.988136943566943, 5.140502156598062)
data <- data.frame(x, y)

# Pearson
cor <- cor_test(data, x="x", y="y", method="pearson")
cor <- cor_test(data, x="x", y="y", method="pearson", alternative="less")
cor <- cor_test(data, x="x", y="y", method="pearson", alternative="greater")
# Robust
cor <- cor_test(data, x="x", y="y", method="spearman")
cor <- cor_test(data, x="x", y="y", method="kendall")
cor <- cor_test(data, x="x", y="y", method="percentage")
cor <- cor_test(data, x="x", y="y", method="biweight")
cor <- cor_test(data, x="x", y="y", method="shepherd")

###############################################################################
# Partial correlation
###############################################################################

library(ppcor)

# Update path to pingouin/datasets/
df <- read.csv("../datasets/partial_corr.csv")

# Partial correlation
pcor.test(x=df$x, y=df$y, z=df[, c("cv1")])
pcor.test(x=df$x, y=df$y, z=df[, c("cv1", "cv2")])
pcor.test(x=df$x, y=df$y, z=df[, c("cv1", "cv2", "cv3")])
pcor.test(x=df$x, y=df$y, z=df[, c("cv1", "cv2", "cv3")], method="spearman")

# Semi partial correlation
# z is removed from the y variable (e.g. y_covar in Pingouin)
spcor.test(x=df$x, y=df$y, z=df$cv1)  # y_covar
spcor.test(x=df$y, y=df$x, z=df$cv1)  # x_covar

spcor.test(x=df$x, y=df$y, z=df[, c("cv1", "cv2")])
spcor.test(x=df$x, y=df$y, z=df[, c("cv1", "cv2", "cv3")])
spcor.test(x=df$y, y=df$x, z=df[, c("cv1", "cv2", "cv3")])  # x_covar
spcor.test(x=df$x, y=df$y, z=df[, c("cv1", "cv2", "cv3")], method="spearman")

