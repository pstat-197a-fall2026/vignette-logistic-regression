# vignette-script.R
# Multiclass (multinomial) logistic regression for red wine quality
#   quality_group: 0 = poor (<= 5), 1 = average (6), 2 = good (>= 7)
#
# Replicates every result and figure in the vignette end to end:
#   model fitting (model_fitting.R) + evaluation (model_evaluations.Rmd)
# HOW TO RUN: open the repo as an RStudio project (or setwd() to the repo
# root), then run source("scripts/vignette-script.R").


#0. SETUP

library(ggplot2)     # plotting (confusion matrix heatmap)
library(nnet)        # multinom(): multinomial logistic regression
library(caret)       # confusionMatrix(): confusion matrix + one-vs-all metrics; train(): cross-validation
library(tidymodels)  # loads dplyr and yardstick (mn_log_loss)
library(readr)       # read_csv(): reading the raw data
library(pROC)        # roc(): ROC curves and AUC

set.seed(123)        # fix the seed so the train/test split is reproducible

# create output folders if they don't exist yet
dir.create("data", showWarnings = FALSE)
dir.create("img",  showWarnings = FALSE)


#1. LOAD AND PREPROCESS THE DATA

# read the raw red wine data (Kaggle/UCI file, comma-separated)
# column names keep their spaces (e.g. "fixed acidity")
wine <- read_csv("data/winequality-red.csv", show_col_types = FALSE)
dim(wine)        # 1599 rows, 12 columns (11 predictors + quality)
head(wine)       # first few rows
str(wine)        # all columns are numeric
summary(wine)    # ranges differ widely across predictors because units differ

colSums(is.na(wine))   # missing values per column: expect all zeros

# change numerical quality scores into three categories
#   0 = poor (score <= 5), 1 = average (score = 6), 2 = good (score >= 7)
wine$quality_group <- ifelse(wine$quality <= 5, 0,
                             ifelse(wine$quality <= 6, 1, 2))
wine$quality_group <- factor(wine$quality_group,
                             levels = c(0, 1, 2))   # store as a factor, not a number
summary(wine$quality)         # raw scores before recoding
summary(wine$quality_group)   # class counts after recoding

# remove the original numeric score so it cannot be used as a predictor
wine$quality <- NULL
str(wine)

# data points in each class: heavily imbalanced, few "good" wines
table(wine$quality_group)
prop.table(table(wine$quality_group))

# save the preprocessed data
write.csv(wine, file = "data/preprocessed_winequality.csv", row.names = FALSE)


#1b. EXPLORATORY ANALYSIS

# duplicate rows: identical measurements that appear more than once
sum(duplicated(wine))

# class balance: number of wines in each quality group
class_balance_plot <- ggplot(wine, aes(x = quality_group)) +
  geom_bar(fill = "steelblue") +
  scale_x_discrete(labels = c("0" = "poor (0)", "1" = "average (1)", "2" = "good (2)")) +
  labs(title = "Number of wines in each quality group",
       x = "Quality group", y = "Count")
class_balance_plot

# distributions of three predictors by quality group
predictor_boxplots <- wine %>%
  select(quality_group, alcohol, `volatile acidity`, sulphates) %>%
  pivot_longer(-quality_group, names_to = "predictor", values_to = "value") %>%
  ggplot(aes(x = quality_group, y = value, fill = quality_group)) +
  geom_boxplot(show.legend = FALSE) +
  facet_wrap(~ predictor, scales = "free_y") +   # each predictor gets its own y-axis
  labs(title = "Selected predictors by quality group",
       x = "Quality group (0 = poor, 1 = average, 2 = good)", y = NULL)
predictor_boxplots


#2. STRATIFIED TRAIN / TEST SPLIT

# sample WITHIN each quality class so that train and test keep the same class
# proportions (important because the classes are imbalanced)
# for each class: hold out 20% (at least 1 row) for testing, train on the rest
train_indices <- unlist(
  lapply(levels(wine$quality_group), function(group) {
    
    group_indices <- which(wine$quality_group == group)   # row numbers in this class
    
    sample(
      group_indices,
      size = length(group_indices) -
        max(1, floor(0.2 * length(group_indices)))        # class size minus the test portion
    )
  })
)

train <- wine[train_indices, ]    # training set: used to fit the model
test  <- wine[-train_indices, ]   # test set: used only for evaluation

# class distributions of the training and test split
table(train$quality_group)
table(test$quality_group)

# set medium quality ("1") as the reference category: every coefficient in the
# model compares poor or good wines against average wines
train$quality_group <- relevel(train$quality_group, ref = "1")

# give the test outcome the same level order as the training outcome
test$quality_group <- factor(test$quality_group, levels = c("1", "0", "2"))

# standardize the predictors (mean 0, SD 1) so coefficients are comparable and
# the optimizer converges reliably. Means and SDs come from the TRAINING set only
# and are applied to both sets, so nothing about the test set leaks into the model.
# Standardizing does not change the model's predictions, only the coefficient scale.
predictors  <- setdiff(names(train), "quality_group")
train_means <- sapply(train[predictors], mean)
train_sds   <- sapply(train[predictors], sd)

train[predictors] <- as.data.frame(scale(train[predictors], center = train_means, scale = train_sds))
test[predictors]  <- as.data.frame(scale(test[predictors],  center = train_means, scale = train_sds))

round(sapply(train[predictors], mean), 3)   # check: all approximately 0

# save the splits so other documents (e.g. the evaluation notebook) can reuse them
save(train, file = "./data/train_data.RData")
save(test,  file = "./data/test_data.RData")


#3. FIT THE MULTINOMIAL LOGISTIC REGRESSION

# multinom() fits two logistic regressions at once (poor vs average, good vs
# average), each with its own coefficients; `.` means "use all other columns"
# maxit = 1000 gives the optimizer enough iterations to fully converge
# (the default of 100 can stop early)
model <- multinom(quality_group ~ ., data = train, maxit = 1000)
model$convergence                 # 0 = converged; 1 = stopped at the iteration limit
summary(model)                    # coefficients and standard errors, one row per non-reference class

exp(coef(model))                  # relative risk ratios: multiplicative change in the odds of
# (poor or good) vs average for a 1-SD increase in the predictor

# Wald tests: z = coefficient / standard error, two-sided p-value from N(0, 1)
z <- summary(model)$coefficients / summary(model)$standard.errors
p <- 2 * pnorm(-abs(z))
round(p, 3)

head(round(fitted(model), 2))     # fitted class probabilities for the first training wines

save(model, file = "./data/multiclass-logistic-model.RData")


#4. PREDICT ON THE TEST SET

# most likely class for each test wine
predicted_classes <- predict(model, newdata = test, type = "class")

# predicted probability of every class for each test wine (each row sums to 1)
# columns follow the factor level order of the training outcome: "1", "0", "2"
predicted_probabilities <- predict(model, newdata = test, type = "probs")
head(predicted_probabilities)

# side-by-side actual vs predicted classes
head(data.frame(Actual = test$quality_group, Predicted = predicted_classes))


#5. CONFUSION MATRIX HEATMAP

# count every (predicted, actual) combination, using level order 0, 1, 2
confusion_matrix <- as.data.frame(table(
  Predicted = factor(predicted_classes,    levels = c("0", "1", "2")),
  Actual    = factor(test$quality_group,   levels = c("0", "1", "2"))
))
confusion_matrix

confusion_matrix_plot <- ggplot(confusion_matrix,
                                aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "white") +                          # one tile per cell
  geom_text(aes(label = Freq)) +                        # print the count in each tile
  scale_fill_gradient(low = "white", high = "blue") +   # darker blue = more wines
  labs(title = "Multinomial Logistic Regression",
       subtitle = "Confusion Matrix",
       x = "Predicted Quality", y = "Actual Quality")
confusion_matrix_plot
ggsave("img/fig-confusion-matrix.png", confusion_matrix_plot,
       width = 6, height = 4.5, dpi = 150)


#6. PREPARE EVALUATION OBJECTS

# factors with level order 0, 1, 2 for caret's confusionMatrix()
test_y <- factor(test$quality_group, levels = c("0", "1", "2"))
test_y_pred <- factor(predicted_classes, levels = c("0", "1", "2"))

summary(test_y)   # heavy imbalance: very few good wines in the test set

# readable class names; the probability columns are renamed by looking up each
# column's class code, so the labels stay correct even though the column order is 1, 0, 2
class_label <- c("0" = "poor", "1" = "avg", "2" = "good")
probs <- as.data.frame(predicted_probabilities)
colnames(probs) <- paste0("pred_prob_", class_label[colnames(probs)])
# columns are now: pred_prob_avg, pred_prob_poor, pred_prob_good

results <- data.frame(
  actual    = factor(recode(as.character(test_y),
                            "0" = "poor", "1" = "average", "2" = "good"),
                     levels = c("average", "poor", "good")),   # same order as the probability columns
  predicted = factor(recode(as.character(test_y_pred),
                            "0" = "poor", "1" = "average", "2" = "good")),
  probs
)
head(results, 25)
save(results, file = "./data/results.RData")


#7. PROBABILITY METRICS: LOG LOSS AND ROC-AUC

# multiclass log loss: penalizes confident wrong predictions (lower is better)
# yardstick matches the probability columns to the truth levels IN ORDER, so the
# columns must be listed as average, poor, good to match levels(results$actual)
mn_log_loss(results, truth = actual,
            pred_prob_avg, pred_prob_poor, pred_prob_good)

# one-vs-all ROC curve for each class, saved as a figure
# (average has the lowest AUC, which is expected: it sits between poor and good)
plot_roc <- function(truth_class, prob_col, label, file) {
  roc_obj <- roc(results$actual == truth_class, results[[prob_col]], quiet = TRUE)
  
  png(file, width = 600, height = 500)
  plot(roc_obj,
       main = paste0("ROC Curve for Wine Quality (", label, " vs All)"),
       xlab = "1 - Specificity",       # legacy.axes = TRUE puts 1 - specificity on the x-axis
       ylab = "Sensitivity",
       col = "#1c7ed6", lwd = 3,
       print.auc = TRUE,               # print the AUC on the plot
       legacy.axes = TRUE,
       cex.main = 0.75)
  abline(a = 0, b = 1, lty = 2, col = "darkgray")   # chance line
  dev.off()
  
  as.numeric(auc(roc_obj))             # return the AUC
}

auc_poor <- plot_roc("poor",    "pred_prob_poor", "Poor",    "img/fig-roc-poor.png")
auc_avg  <- plot_roc("average", "pred_prob_avg",  "Average", "img/fig-roc-average.png")
auc_good <- plot_roc("good",    "pred_prob_good", "Good",    "img/fig-roc-good.png")

c(poor = auc_poor, average = auc_avg, good = auc_good)   # AUC per class
mean(c(auc_poor, auc_avg, auc_good))                     # macro-averaged AUC


#8. CLASSIFICATION METRICS

# confusion matrix plus overall accuracy and one-vs-all metrics for each class
# (sensitivity, specificity, precision, recall, F1, ...)
cm <- confusionMatrix(test_y_pred, test_y, mode = "everything")
cm


#8b. CLASS WEIGHTS FOR CLASS IMBALANCE

# good wines are rare, so the unweighted model rarely predicts "good".
# weight each training wine inversely to its class size, w = n / (3 * n_class),
# so all three classes contribute equally to the likelihood
counts <- table(train$quality_group)
w <- as.numeric(nrow(train) / (3 * counts[train$quality_group]))

model_w <- multinom(quality_group ~ ., data = train, weights = w,
                    maxit = 1000, trace = FALSE)
model_w$convergence               # 0 = converged

# predictions and metrics for the weighted model (level order 0, 1, 2 as above)
pred_w <- factor(predict(model_w, newdata = test, type = "class"),
                 levels = c("0", "1", "2"))
cm_w <- confusionMatrix(pred_w, test_y, mode = "everything")

# recall by class: weighting should raise recall for good wines
data.frame(class             = c("poor", "average", "good"),
           recall_unweighted = cm$byClass[, "Recall"],
           recall_weighted   = cm_w$byClass[, "Recall"])

# overall accuracy and macro F1: weighting usually costs a little accuracy
data.frame(model    = c("unweighted", "weighted"),
           accuracy = c(cm$overall["Accuracy"], cm_w$overall["Accuracy"]),
           macro_f1 = c(mean(cm$byClass[, "F1"]), mean(cm_w$byClass[, "F1"])))


#9. MACRO, WEIGHTED AND MICRO AVERAGES

# class weights: the share of test wines in each class (for the weighted average)
weights <- as.numeric(prop.table(table(test_y)))   # order: class 0, 1, 2

# per-class metrics from caret, one column per class
#   macro average    = plain mean across the three classes (every class counts equally)
#   weighted average = mean weighted by class size (large classes count more)
averaging_scores <- data.frame(t(cm$byClass)) %>%
  setNames(c("Poor", "Average", "Good")) %>%
  mutate(macro_avg    = (Poor + Average + Good) / 3,
         weighted_avg = weights[1] * Poor + weights[2] * Average + weights[3] * Good)

# micro average: pool counts across classes first, then compute the metric.
# if a metric for class i is a_i / b_i, the micro average is sum(a_i) / sum(b_i)
# in cm$table, rows = predicted class and columns = actual class
tab <- cm$table
n   <- sum(tab)                                          # number of test wines
tp  <- sum(diag(tab))                                    # total true positives
tn  <- sum(sapply(1:3, function(i) sum(tab[-i, -i])))    # total true negatives
neg <- sum(sapply(1:3, function(i) sum(tab[, -i])))      # total actual negatives (TN + FP)
pred_pos <- sum(sapply(1:3, function(i) sum(tab[i, ])))  # total predicted positives (TP + FP)
pred_neg <- sum(sapply(1:3, function(i) sum(tab[-i, ]))) # total predicted negatives (TN + FN)

micro_sensitivity <- tp / n                              # TP / (TP + FN), equals accuracy
micro_specificity <- tn / neg                            # TN / (TN + FP)
micro_pos_pred    <- tp / pred_pos                       # TP / (TP + FP)
micro_neg_pred    <- tn / pred_neg                       # TN / (TN + FN)
micro_precision   <- micro_pos_pred                      # same as positive predictive value
micro_recall      <- micro_sensitivity                   # same as sensitivity
micro_f1          <- 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
micro_prevalence          <- 1 / 3                       # always 1/3 with three classes
micro_detection_rate      <- tp / (3 * n)
micro_detection_prevalence <- pred_pos / (3 * n)         # also always 1/3
micro_balanced_accuracy   <- (micro_sensitivity + micro_specificity) / 2

# same row order as cm$byClass
averaging_scores$micro_avg <- c(micro_sensitivity, micro_specificity,
                                micro_pos_pred, micro_neg_pred,
                                micro_precision, micro_recall, micro_f1,
                                micro_prevalence, micro_detection_rate,
                                micro_detection_prevalence, micro_balanced_accuracy)
averaging_scores
write.csv(averaging_scores, "data/averaging-scores.csv")


#10. MODEL VALIDATION

# (a) baseline: accuracy of always predicting the most common class
baseline_acc <- unname(cm$overall["AccuracyNull"])
model_acc    <- unname(cm$overall["Accuracy"])
data.frame(model = c("Always predict majority class", "Multiclass logistic regression"),
           accuracy = c(baseline_acc, model_acc))
cm$overall["AccuracyPValue"]   # one-sided test: is the model better than the baseline?

# (b) 10-fold cross-validation on the training data, also tuning `decay`
#     (the ridge / L2 penalty in multinom; decay = 0 is the unpenalized model above)
train_x <- dplyr::select(train, -quality_group)
train_y <- factor(train$quality_group, levels = c("0", "1", "2"))
test_x  <- dplyr::select(test,  -quality_group)

set.seed(123)                                          # reproducible fold assignment
cv_fit <- train(
  x = train_x, y = train_y,
  method     = "multinom",
  preProcess = c("center", "scale"),                   # scaling done inside each fold (no leakage)
  trControl  = trainControl(method = "cv", number = 10),
  tuneGrid   = expand.grid(decay = c(0, 0.001, 0.01, 0.1, 1)),
  trace      = FALSE                                   # silence multinom's iteration output
)
cv_fit$results                                         # CV accuracy and Kappa for each decay value

png("img/fig-cv-decay.png", width = 600, height = 450)
plot(cv_fit)                                           # CV accuracy vs. decay
dev.off()

# test-set performance of the cross-validated model, for comparison with the original fit
cv_cm <- confusionMatrix(predict(cv_fit, test_x), test_y, mode = "everything")
cv_cm$overall["Accuracy"]


#11. SESSION INFO

# documents package versions for reproducibility
sessionInfo()