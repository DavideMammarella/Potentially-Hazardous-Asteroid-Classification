# Title     : Potentially Hazardous Asteroids: A classification performed on the NASA dataset
# Objective : Classifying Potentially Hazardous Asteroids, a project for the Machine Learning course of the University of Milano-Bicocca
# Created by: Mammarella Davide
# Created on: 21/11/2020

# all necessary packages are installed
install.packages("here")
install.packages("FactoMineR")
install.packages("factoextra")
install.packages("rpart")
install.packages("rattle")
install.packages("e1071")
install.packages("pROC")
install.packages("caret")

# all necessary packages are called
library("here")
library("FactoMineR")
library("factoextra")
library("rpart")
library("rattle")
library("e1071")
library("pROC")
library("caret")

#----------------------------------------------------------------------------------------
#	Exploratory Analysis
#----------------------------------------------------------------------------------------

# dataset import
dataset <- read.csv(here("dataset", "nasa.csv"), header = TRUE)
cat("\n ================================================== \n Dataset: \n \n")
str(dataset)
# number of instances
instances.number <- nrow(dataset)
# number of features
features.number <- ncol(dataset)
# null values in the dataset
nullvalue.present <- sum(is.na(dataset))
cat("\n ================================================== \n [Dataset] Null values: \n")
cat(nullvalue.present, "\n")
# hazardous asteroids in the dataset
num.hazardous.instances <- sum(dataset["Hazardous"] == "True")
num.nonhazardous.instances <- instances.number - num.hazardous.instances
barplot(c(num.hazardous.instances, num.nonhazardous.instances), names.arg = c("Hazardous", "Non-Hazardous"),
         col = c("#BB6364", "#64BB63"), ylim = c(0, 5000))
# columns containing a single non-discriminating value
not.Discriminative.Values <- dataset[, c("Orbiting.Body", "Equinox")]
cat("\n ================================================== \n [Dataset] Not discriminative values: \n")
unique(not.Discriminative.Values)
# delete columns considered inactive or containing a single non-discriminating value
columns.to.drop <- c("Neo.Reference.ID", "Name", "Close.Approach.Date", "Epoch.Date.Close.Approach", "Orbiting.Body", "Orbit.ID", "Orbit.Determination.Date", "Equinox", "Absolute.Magnitude", "Minimum.Orbit.Intersection")
bool.columns.to.drop <- !names(dataset) %in% columns.to.drop
dataset.light <- dataset[, bool.columns.to.drop]
dataset.active <- subset(dataset.light, select = -Hazardous)

#----------------------------------------------------------------------------------------
#	Principal Component Analysis
#----------------------------------------------------------------------------------------

# compute PCA
res.pca <- PCA(dataset.active, graph = FALSE)
# print eigenvalues and percentage of variance associated
cat("\n ================================================== \n [PCA] Eigenvalues and % of variance associated: \n \n")
get_eigenvalue(res.pca)
# bar plot relation between percentage of evariances and dimensions
fviz_eig(res.pca, addlabels = TRUE, ylim = c(0, 32))

# Analysis on variables -----------------------------------------------------------------

# var from res.pca
var <- get_pca_var(res.pca)
# correlations between variables and dimensions
cat("\n ================================================== \n [PCA] Correlations between variables and dimension: \n \n")
var$cor
# plot correlations between variables and dimensions (considering also the cos2 component)
fviz_pca_var(res.pca, geom = c("point", "text"), col.var = "cos2", gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), repel = TRUE, axes = c(1, 2))
fviz_cos2(res.pca, choice = "var", axes = 1:5)

# Analysis on individuals ---------------------------------------------------------------

# ind from res.pca
ind <- get_pca_ind(res.pca)
ind.cos2.dimensions <- ind$cos2[, 1:5]
individuals.orderedby.dim1 <- ind.cos2.dimensions[order(ind.cos2.dimensions[, 1], decreasing = TRUE),]
individuals.orderedby.dim2 <- ind.cos2.dimensions[order(ind.cos2.dimensions[, 2], decreasing = TRUE),]
individuals.orderedby.dim3 <- ind.cos2.dimensions[order(ind.cos2.dimensions[, 3], decreasing = TRUE),]
individuals.orderedby.dim4 <- ind.cos2.dimensions[order(ind.cos2.dimensions[, 4], decreasing = TRUE),]
individuals.orderedby.dim5 <- ind.cos2.dimensions[order(ind.cos2.dimensions[, 5], decreasing = TRUE),]

# Dataset cleaning ----------------------------------------------------------------------

dataset.clean <- dataset.light[rbind(rownames(individuals.orderedby.dim1)[1:500],
                                     rownames(individuals.orderedby.dim2)[1:500],
                                     rownames(individuals.orderedby.dim3)[1:500],
                                     rownames(individuals.orderedby.dim4)[1:500],
                                     rownames(individuals.orderedby.dim5)[1:500]),]
dataset.clean <- dataset.clean[!duplicated(dataset.clean),]

columns.to.drop.after.analysis <- c("Est.Dia.in.M.min.", "Est.Dia.in.M.max.", "Est.Dia.in.Miles.min.",
                                    "Est.Dia.in.Miles.max.", "Est.Dia.in.Feet.min.", "Est.Dia.in.Feet.max.",
                                    "Miles.per.hour", "Relative.Velocity.km.per.hr", "Miss.Dist..miles.",
                                    "Miss.Dist..lunar.", "Miss.Dist..Astronomical.", "Semi.Major.Axis",
                                    "Aphelion.Dist", "Asc.Node.Longitude", "Mean.Anomaly", "Perihelion.Arg",
                                    "Epoch.Osculation")

bool.columns.to.drop.after.analysis <- !names(dataset.light) %in% columns.to.drop.after.analysis
dataset.clean <- dataset.clean[, bool.columns.to.drop.after.analysis]
dataset.clean$Hazardous <- as.factor(dataset.clean$Hazardous)
cat("\n ================================================== \n [PCA] Cleaned dataset: \n \n")
str(dataset.clean)

#----------------------------------------------------------------------------------------
#	Train and Test datasets
#----------------------------------------------------------------------------------------

# split dataset (70% train, 30% test)
split.dataset <- function(dataset.clean, p = 0.70, s = 1) {
  set.seed(s)
  index <- sample(1:dim(dataset.clean)[1])
  train <- dataset.clean[index[1:floor(dim(dataset.clean)[1] * p)],]
  test <- dataset.clean[index[((ceiling(dim(dataset.clean)[1] * p)) + 1):dim(dataset.clean)[1]],]
  return(list(train = train, test = test)) }

# verify dataset split
all.set <- split.dataset(dataset.clean, p = 0.70)
train.set <- all.set$train
test.set <- all.set$test

# print hazardous values references for analysis
cat("\n ================================================== \n [Dataset] Hazardous values distribution:")
prop.table(table(dataset$Hazardous))
cat("\n ================================================== \n [Clean Dataset] Hazardous values distribution:")
prop.table(table(dataset.clean$Hazardous))
cat("\n ================================================== \n [Train dataset] Hazardous values distribution after split:")
prop.table(table(train.set$Hazardous))
cat("\n ================================================== \n [Train dataset] Hazardous values after split:")
prop.table(table(test.set$Hazardous))


#----------------------------------------------------------------------------------------
#	Baseline Model
#----------------------------------------------------------------------------------------

# assume that every asteroid in the test set is not hazardous
test.set$Prediction <- 0
# print confusion matrix and accuracy
cat("\n ================================================== \n [Baseline Model] Confusion Matrix: \n")
confusion.matrix <- table(pred = test.set$Prediction, true = test.set$Hazardous)
confusion.matrix
cat("\n ================================================== \n [Baseline Model] Accuracy: \n")
cat(sum(diag(confusion.matrix)) / sum(confusion.matrix), "\n")
# reset test set
test.set <- subset(test.set, select = -Prediction)

#----------------------------------------------------------------------------------------
#	Decision Tree Model
#----------------------------------------------------------------------------------------

# Basic decision tree -------------------------------------------------------------------

# train
decisionTree <- rpart(Hazardous ~ ., data = train.set, method = "class")
# plot
fancyRpartPlot(decisionTree, main = "Decision Tree")
# prediction
decision.tree.pred <- predict(decisionTree, test.set, type = "class")
# print confusion matrix and accuracy
cat("\n ================================================== \n [Decision Tree Model] Confusion Matrix: \n")
confusion.matrix.tree <- table(pred = decision.tree.pred, true = test.set$Hazardous)
confusion.matrix.tree
cat("\n ================================================== \n [Decision Tree Model] Accuracy: \n")
cat(sum(diag(confusion.matrix.tree)) / sum(confusion.matrix.tree), "\n")

# Complexity parameter ------------------------------------------------------------------

# plot cp for analysis
plotcp(decisionTree)

# Pruned decision tree ------------------------------------------------------------------

# prune
prunedDecisionTree <- prune(decisionTree, cp = 0.031)
# plot
fancyRpartPlot(prunedDecisionTree, main = "Pruned Decision Tree")
# prediction
pruned.decision.tree.pred <- predict(prunedDecisionTree, test.set, type = "class")
# print confusion matrix and accuracy
cat("\n ================================================== \n [Pruned Decision Tree Model] Confusion Matrix: \n")
confusion.matrix.pruned.tree <- table(pred = pruned.decision.tree.pred, true = test.set$Hazardous)
confusion.matrix.pruned.tree
cat("\n ================================================== \n [Pruned Decision Tree Model] Accuracy: \n")
cat(sum(diag(confusion.matrix.pruned.tree)) / sum(confusion.matrix.pruned.tree), "\n")

#----------------------------------------------------------------------------------------
#	Support Vector Machine Model
#----------------------------------------------------------------------------------------

# tune (uncomment only on tuning phase!)
#tuned <- tune.svm(Hazardous ~ ., data = dataset.clean, kernel='radial', type = "C-classification",
#                  cost= c(0.1, 10, 50), gamma = c(0.01, 0.4, 3))
#summary(tuned)
#plot(tuned)
# train
svm.model <- svm(Hazardous ~ ., data = train.set, kernel = 'radial', type = "C-classification", cost = 10, gamma = 0.4)
# prediction
svm.pred <- predict(svm.model, test.set)
# print accuracy
cat("\n ================================================== \n [SVM Model] Confusion Matrix: \n")
confusion.matrix.svm <- table(pred = svm.pred, true = test.set$Hazardous)
confusion.matrix.svm
cat("\n ================================================== \n [SVM Model] Accuracy: \n")
cat(sum(diag(confusion.matrix.svm)) / sum(confusion.matrix.svm), "\n")

#----------------------------------------------------------------------------------------
#	Experiments
#----------------------------------------------------------------------------------------

# Decision Tree -------------------------------------------------------------------------

# 10-fold cross validation
folds <- cut(seq(1, nrow(dataset.clean)), breaks = 10, labels = FALSE)
confusion.matrix.decision.tree.overall.list <- vector(mode = "list", length = 10)
test.set.overall <- NULL
predict.overall.decision.tree <- NULL
cat("\n ================================================== \n [Decision Tree Model] 10-fold cross validation, distribution of Hazardous: \n")
time.tree <- Sys.time()
for (i in 1:10) {
  testIndexes <- which(folds == i, arr.ind = TRUE)
  test.set.ten.fold <- dataset.clean[testIndexes,]
  train.set.ten.fold <- dataset.clean[-testIndexes,]
  # we show that the percentage distribution on PHA is kept similar to dataset.clean
  print(prop.table(table(train.set.ten.fold$Hazardous)))
  print(prop.table(table(test.set.ten.fold$Hazardous)))
  decision.tree.model.fold <- rpart(Hazardous ~ ., data = train.set.ten.fold, method = "class")
  decision.tree.pred.fold <- predict(decision.tree.model.fold, test.set.ten.fold, type = "class")
  decision.tree.pred.fold.prob <- predict(decision.tree.model.fold, test.set.ten.fold, type = "prob")
  predict.overall.decision.tree <- append(predict.overall.decision.tree, decision.tree.pred.fold.prob[, 2])
  confusion.matrix.decision.tree.single <- table(pred = decision.tree.pred.fold, true = test.set.ten.fold$Hazardous)
  confusion.matrix.decision.tree.overall.list[[i]] <- confusion.matrix.decision.tree.single
}
time.tree <- Sys.time() - time.tree
confusion.matrix.decision.tree.overall <- confusion.matrix.decision.tree.overall.list[[1]]

for (i in 2:10) {
  confusion.matrix.decision.tree.overall <- confusion.matrix.decision.tree.overall + confusion.matrix.decision.tree.overall.list[[i]]
}

cat("\n ================================================== \n [Decision Tree Model] Confusion Matrix evaluated with 10-fold cross validation: \n")
confusion.matrix.decision.tree.overall
cat("\n ================================================== \n [Decision Tree Model] Accuracy evaluated with 10-fold cross validation: \n")
cat(sum(diag(confusion.matrix.decision.tree.overall)) / sum(confusion.matrix.decision.tree.overall), "\n")

# precision
cat("\n ================================================== \n [Decision Tree Model] Precision (True): \n")
precision.decision.tree.T <- confusion.matrix.decision.tree.overall[2, 2] / (confusion.matrix.decision.tree.overall[2, 2] + confusion.matrix.decision.tree.overall[2, 1])
cat(precision.decision.tree.T, "\n")
cat("\n ================================================== \n [Decision Tree Model] Precision (False): \n")
precision.decision.tree.F <- confusion.matrix.decision.tree.overall[1, 1] / (confusion.matrix.decision.tree.overall[1, 2] + confusion.matrix.decision.tree.overall[1, 1])
cat(precision.decision.tree.F, "\n")

# recall
cat("\n ================================================== \n [Decision Tree Model] Recall (True): \n")
recall.decision.tree.T <- confusion.matrix.decision.tree.overall[2, 2] / (confusion.matrix.decision.tree.overall[2, 2] + confusion.matrix.decision.tree.overall[1, 2])
cat(recall.decision.tree.T, "\n")
cat("\n ================================================== \n [Decision Tree Model] Recall (False): \n")
recall.decision.tree.F <- confusion.matrix.decision.tree.overall[1, 1] / (confusion.matrix.decision.tree.overall[1, 1] + confusion.matrix.decision.tree.overall[2, 1])
cat(recall.decision.tree.F, "\n")

# f-measure
cat("\n ================================================== \n [Decision Tree Model] F-Measure (True): \n")
f.measure.T.tree <- (2 *
  precision.decision.tree.T *
  recall.decision.tree.T) / (precision.decision.tree.T + recall.decision.tree.T)
cat(f.measure.T.tree, "\n")
cat("\n ================================================== \n [Decision Tree Model] F-Measure (False): \n")
f.measure.F.tree <- (2 *
  precision.decision.tree.F *
  recall.decision.tree.F) / (precision.decision.tree.F + recall.decision.tree.F)
cat(f.measure.F.tree, "\n")

#ROC and AUC
par(pty = "s")
decision.tree.ROC <- roc(dataset.clean$Hazardous ~ predict.overall.decision.tree, plot = TRUE, print.auc = TRUE, col = "black", lwd = 4, legacy.axes = TRUE, main = "Decision Tree ROC")

# SVM -----------------------------------------------------------------------------------

# 10-fold cross validation
folds <- cut(seq(1, nrow(dataset.clean)), breaks = 10, labels = FALSE)
confusion.matrix.svm.overall.list <- vector(mode = "list", length = 10)
test.set.overall <- NULL
predict.overall.svm <- NULL
cat("\n ================================================== \n [SVM Model] 10-fold cross validation, distribution of Hazardous: \n")
time.svm <- Sys.time()
for (i in 1:10) {
  testIndexes <- which(folds == i, arr.ind = TRUE)
  test.set.ten.fold <- dataset.clean[testIndexes,]
  train.set.ten.fold <- dataset.clean[-testIndexes,]
  # we show that the percentage distribution on PHA is kept similar to dataset.clean
  print(prop.table(table(train.set.ten.fold$Hazardous)))
  print(prop.table(table(test.set.ten.fold$Hazardous)))
  svm.model.fold <- svm(Hazardous ~ ., data = train.set.ten.fold, kernel = 'radial', type = "C-classification", cost = 10, gamma = 0.4, probability = TRUE)
  svm.pred.fold <- predict(svm.model.fold, test.set.ten.fold)
  svm.pred.fold.prob <- predict(svm.model.fold, test.set.ten.fold, probability = TRUE)
  predict.overall.svm <- append(predict.overall.svm, attr(svm.pred.fold.prob, "probabilities")[, 2])
  confusion.matrix.svm.single <- table(pred = svm.pred.fold, true = test.set.ten.fold$Hazardous)
  confusion.matrix.svm.overall.list[[i]] <- confusion.matrix.svm.single
}
time.svm <- Sys.time() - time.svm
confusion.matrix.svm.overall <- confusion.matrix.svm.overall.list[[1]]

for (i in 2:10) {
  confusion.matrix.svm.overall <- confusion.matrix.svm.overall + confusion.matrix.svm.overall.list[[i]]
}

cat("\n ================================================== \n [SVM Model] Confusion Matrix evaluated with 10-fold cross validation: \n")
confusion.matrix.svm.overall
cat("\n ================================================== \n [SVM Model] Accuracy evaluated with 10-fold cross validation: \n")
cat(sum(diag(confusion.matrix.svm.overall)) / sum(confusion.matrix.svm.overall), "\n")

# precision
cat("\n ================================================== \n [SVM Model] Precision (True): \n")
precision.svm.T <- confusion.matrix.svm.overall[2, 2] / (confusion.matrix.svm.overall[2, 2] + confusion.matrix.svm.overall[2, 1])
cat(precision.svm.T, "\n")
cat("\n ================================================== \n [SVM Model] Precision (False): \n")
precision.svm.F <- confusion.matrix.svm.overall[1, 1] / (confusion.matrix.svm.overall[1, 2] + confusion.matrix.svm.overall[1, 1])
cat(precision.svm.F, "\n")

# recall
cat("\n ================================================== \n [SVM Model] Recall (True): \n")
recall.svm.T <- confusion.matrix.svm.overall[2, 2] / (confusion.matrix.svm.overall[2, 2] + confusion.matrix.svm.overall[1, 2])
cat(recall.svm.T, "\n")
cat("\n ================================================== \n [SVM Model] Recall (False): \n")
recall.svm.F <- confusion.matrix.svm.overall[1, 1] / (confusion.matrix.svm.overall[1, 1] + confusion.matrix.svm.overall[2, 1])
cat(recall.svm.F, "\n")

# f-measure
cat("\n ================================================== \n [SVM Model] F-Measure (True): \n")
f.measure.T.svm <- (2 * precision.svm.T * recall.svm.T) / (precision.svm.T + recall.svm.T)
cat(f.measure.T.svm, "\n")
cat("\n ================================================== \n [SVM Model] F-Measure (False): \n")
f.measure.F.svm <- (2 * precision.svm.F * recall.svm.F) / (precision.svm.F + recall.svm.F)
cat(f.measure.F.svm, "\n")

# ROC and AUC
par(pty = "s")
svm.ROC <- roc(dataset.clean$Hazardous ~ predict.overall.svm, plot = TRUE, print.auc = TRUE, col = "black", lwd = 4, legacy.axes = TRUE, main = "SVM ROC")

# Decision Tree and SVM comparison on ROC and AUC ---------------------------------------

decision.tree.ROC <- roc(dataset.clean$Hazardous ~ predict.overall.decision.tree, plot = TRUE, print.auc = TRUE, col = "#64BB63", lwd = 4, legacy.axes = TRUE, main = "ROC Curves")
svm.ROC <- roc(dataset.clean$Hazardous ~ predict.overall.svm, plot = TRUE, print.auc = TRUE, col = "#386EA5", lwd = 4, print.auc.y = 0.4, add = TRUE, legacy.axes = TRUE)
legend("bottom", legend = c("Decision Tree", "SVM"), col = c("#64BB63", "#386EA5"), lty = c(1,1), bty = "n")
cis <- rbind(ci(svm.ROC), ci(decision.tree.ROC))
row.names(cis) <- c("SVM", "Decision tree")
dotplot(cis, main = "ROC", pch = 16, cex=2, col = "#386EA5")