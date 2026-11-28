# vignette-script.R
# Multinomial logistic regression for red wine quality
#   quality_group: 0 = poor (<= 5), 1 = average (6), 2 = good (>= 7)
#
# Replicates every result and figure in the vignette end to end, in the same
# order. The section numbers below match the section numbers in the vignette.
#
# HOW TO RUN: open the repo as an RStudio project (or setwd() to the repo root), then run
#   source("scripts/vignette-script.R", echo = TRUE)
# echo = TRUE prints every result to the console; plain source() would print nothing.
# Figures are saved to img/.


# 3. Setup

library(ggplot2)     # plotting
library(nnet)        # multinom(): multinomial logistic regression
library(caret)       # confusionMatrix(): confusion matrix + one-vs-all metrics; train(): cross-validation
library(tidymodels)  # loads dplyr, tidyr and yardstick (mn_log_loss)
library(readr)       # read_csv(): reading the raw data
library(pROC)        # roc(): ROC curves and AUC

# create the output folder for figures if it doesn't exist yet
dir.create("img", showWarnings = FALSE)

# read the raw red wine data (Kaggle/UCI file, comma-separated)
# column names keep their spaces (e.g. "fixed acidity")
wine <- read_csv("data/winequality-red.csv", show_col_types = FALSE)
dim(wine)        # 1599 rows, 12 columns (11 predictors + quality)
head(wine)       # first few rows
str(wine)        # all columns are numeric
summary(wine)    # ranges differ widely across predictors because units differ


# 3.1 Defining the outcome

# change numerical quality scores into three categories
#   0 = poor (score <= 5), 1 = average (score = 6), 2 = good (score >= 7)
wine$quality_group <- ifelse(wine$quality <= 5, 0,
                             ifelse(wine$quality <= 6, 1, 2))
wine$quality_group <- factor(wine$quality_group,
                             levels = c(0, 1, 2))   # store as a factor, not a number

# remove the raw score so it can't be used as a predictor
wine$quality <- NULL
str(wine)

# number and share of wines in each class: heavily imbalanced, few "good" wines
table(wine$quality_group)
prop.table(table(wine$quality_group))

# save the preprocessed data
write.csv(wine, file = "data/preprocessed_winequality.csv", row.names = FALSE)


# 4. Exploratory analysis

# 4.1 Missing data and duplicates
colSums(is.na(wine))    # missing values per column: expect all zeros
sum(duplicated(wine))   # rows that repeat an earlier row

# 4.2 Class balance
class_balance_plot <- ggplot(wine, aes(x = quality_group)) +
  geom_bar(fill = "steelblue") +
  scale_x_discrete(labels = c("0" = "poor (0)", "1" = "average (1)", "2" = "good (2)")) +
  labs(title = "Number of wines in each quality group",
       x = "Quality group", y = "Count")
class_balance_plot
ggsave("img/fig-class-balance.png", class_balance_plot, width = 6, height = 3.5, dpi = 150)

round(100 * mean(wine$quality_group == "2"), 1)   # percent of good wines

# 4.3 Predictors by class: distributions of three predictors by quality group
predictor_boxplots <- wine %>%
  select(quality_group, alcohol, `volatile acidity`, sulphates) %>%
  pivot_longer(-quality_group, names_to = "predictor", values_to = "value") %>%
  ggplot(aes(x = quality_group, y = value, fill = quality_group)) +
  geom_boxplot(show.legend = FALSE) +
  facet_wrap(~ predictor, scales = "free_y") +   # each predictor gets its own y-axis
  labs(title = "Selected predictors by quality group",
       x = "Quality group (0 = poor, 1 = average, 2 = good)", y = NULL)
predictor_boxplots
ggsave("img/fig-predictor-boxplots.png", predictor_boxplots, width = 8, height = 4, dpi = 150)


# 5. Data partitioning

# 5.1 Stratified train/test split
# sample WITHIN each quality class so that train and test keep the same class
# proportions (important because the classes are imbalanced)
set.seed(123)   # makes the split reproducible

# for each class: hold out 20% (at least 1 row) for testing, train on the rest
train_indices <- unlist(
  lapply(levels(wine$quality_group), function(group) {
    group_indices <- which(wine$quality_group == group)   # row numbers in this class
    sample(group_indices,
           size = length(group_indices) -
             max(1, floor(0.2 * length(group_indices))))  # class size minus the test portion
  })
)

train <- wine[train_indices, ]    # training set: used to fit the model
test  <- wine[-train_indices, ]   # test set: used only for evaluation

table(train$quality_group)
table(test$quality_group)

# 5.2 Choosing the reference class
# set average ("1") as the reference category: every coefficient in the model
# compares poor or good wines against average wines
train$quality_group <- relevel(train$quality_group, ref = "1")

# give the test outcome the same level order as the training outcome
test$quality_group <- factor(test$quality_group, levels = c("1", "0", "2"))

levels(train$quality_group)

# 5.3 Standardizing the predictors
# rescale each predictor to mean 0 and SD 1 so coefficients are comparable and
# the optimizer converges reliably. Means and SDs come from the TRAINING set only
# and are applied to both sets, so nothing about the test set leaks into the model.
# Standardizing does not change the model's predictions, only the coefficient scale.
predictors  <- setdiff(names(train), "quality_group")
train_means <- sapply(train[predictors], mean)
train_sds   <- sapply(train[predictors], sd)

train[predictors] <- as.data.frame(scale(train[predictors], center = train_means, scale = train_sds))
test[predictors]  <- as.data.frame(scale(test[predictors],  center = train_means, scale = train_sds))

round(sapply(train[predictors], mean), 3)   # check: all approximately 0

# 6. From binary to multinomial
# (no code: this section is the math in the vignette)

# 7. Fitting the model

# multinom() fits two logistic regressions at once (poor vs average, good vs
# average), each with its own coefficients; `.` means "use all other columns"
# maxit = 1000 gives the optimizer enough iterations to fully converge
# (the default of 100 can stop early)
model <- multinom(quality_group ~ ., data = train, maxit = 1000)
model$convergence   # 0 = converged; 1 = stopped at the iteration limit
summary(model)      # coefficients and standard errors, one row per non-reference class

# 7.1 Interpreting coefficients
# relative risk ratios: multiplicative change in the odds of (poor or good) vs
# average for a 1-SD increase in the predictor
round(exp(coef(model)), 2)

# Wald tests: z = coefficient / standard error, two-sided p-value from N(0, 1)
z <- summary(model)$coefficients / summary(model)$standard.errors
p <- 2 * pnorm(-abs(z))
round(p, 3)

# SD of alcohol in the original units (used to describe a one-SD change in the text)
round(train_sds["alcohol"], 2)


# 8. Predictions

# most likely class for each test wine
predicted_classes <- predict(model, newdata = test, type = "class")

# predicted probability of every class for each test wine
# columns follow the factor level order of the training outcome: "1", "0", "2"
predicted_probabilities <- predict(model, newdata = test, type = "probs")

round(head(predicted_probabilities), 3)
head(data.frame(Actual = test$quality_group, Predicted = predicted_classes))

# each row of probabilities sums to 1
summary(rowSums(predicted_probabilities))


# 9. Evaluating the model

# 9.1 Confusion matrix
# count every (predicted, actual) pair, with levels in the order 0, 1, 2
cm_counts <- table(
  Predicted = factor(predicted_classes,  levels = c("0", "1", "2")),
  Actual    = factor(test$quality_group, levels = c("0", "1", "2"))
)
cm_counts
confusion_matrix <- as.data.frame(cm_counts)

confusion_matrix_plot <- ggplot(confusion_matrix, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "white") +                          # one tile per cell
  geom_text(aes(label = Freq)) +                        # count in each tile
  scale_fill_gradient(low = "white", high = "blue") +   # darker = more wines
  labs(title = "Multinomial Logistic Regression",
       subtitle = "Confusion Matrix",
       x = "Predicted Quality", y = "Actual Quality")
confusion_matrix_plot
ggsave("img/fig-confusion-matrix.png", confusion_matrix_plot, width = 6, height = 4.5, dpi = 150)

# 9.2 Accuracy and per-class metrics
# factors with level order 0, 1, 2, as confusionMatrix() expects
test_y      <- factor(test$quality_group, levels = c("0", "1", "2"))
test_y_pred <- factor(predicted_classes,  levels = c("0", "1", "2"))

# overall accuracy plus one-vs-all metrics for each class
cm <- confusionMatrix(test_y_pred, test_y, mode = "everything")
cm

cm$overall[c("Accuracy", "AccuracyNull")]     # AccuracyNull = always guess the largest class
round(cm$byClass[, c("Precision", "Recall", "F1")], 3)

# 9.3 Macro, weighted and micro averages
# share of test wines in each class, in the order 0, 1, 2
weights <- as.numeric(prop.table(table(test_y)))

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
micro_prevalence           <- 1 / 3                      # always 1/3 with three classes
micro_detection_rate       <- tp / (3 * n)
micro_detection_prevalence <- pred_pos / (3 * n)         # also always 1/3
micro_balanced_accuracy    <- (micro_sensitivity + micro_specificity) / 2

# same row order as cm$byClass
averaging_scores$micro_avg <- c(micro_sensitivity, micro_specificity,
                                micro_pos_pred, micro_neg_pred,
                                micro_precision, micro_recall, micro_f1,
                                micro_prevalence, micro_detection_rate,
                                micro_detection_prevalence, micro_balanced_accuracy)
averaging_scores
round(averaging_scores[c("Precision", "Recall", "F1"), ], 3)   # the rows shown in the vignette
write.csv(averaging_scores, "data/averaging-scores.csv")

# 9.4 ROC curves and log loss
# readable class names; the probability columns are renamed by looking up each
# column's class code, so the labels stay correct even though the column order is 1, 0, 2
class_label <- c("0" = "poor", "1" = "avg", "2" = "good")
probs <- as.data.frame(predicted_probabilities)
colnames(probs) <- paste0("pred_prob_", class_label[colnames(probs)])
# columns are now: pred_prob_avg, pred_prob_poor, pred_prob_good

results <- data.frame(
  actual = factor(recode(as.character(test_y),
                         "0" = "poor", "1" = "average", "2" = "good"),
                  levels = c("average", "poor", "good")),   # same order as the probability columns
  probs
)
head(results)

# one-vs-all ROC curve: e.g. poor vs. not poor, ranked by the predicted probability of poor
plot_roc <- function(truth_class, prob_col, label) {
  roc_obj <- roc(results$actual == truth_class, results[[prob_col]], quiet = TRUE)
  plot(roc_obj,
       main = paste0(label, " vs. all"),
       col = "#1c7ed6", lwd = 3,
       print.auc = TRUE, print.auc.cex = 1.3,   # print the AUC on the plot
       legacy.axes = TRUE)                      # x-axis = 1 - specificity; the gray line is chance
  as.numeric(auc(roc_obj))                      # return the AUC
}

png("img/fig-roc.png", width = 1000, height = 380)   # save the three curves side by side
par(mfrow = c(1, 3))
auc_poor <- plot_roc("poor",    "pred_prob_poor", "Poor")
auc_avg  <- plot_roc("average", "pred_prob_avg",  "Average")
auc_good <- plot_roc("good",    "pred_prob_good", "Good")
par(mfrow = c(1, 1))
dev.off()

round(c(poor = auc_poor, average = auc_avg, good = auc_good), 3)   # AUC per class

# multiclass log loss: penalizes confident wrong predictions (lower is better)
# yardstick matches the probability columns to the truth levels IN ORDER, so the
# columns must be listed as average, poor, good to match levels(results$actual)
mn_log_loss(results, truth = actual,
            pred_prob_avg, pred_prob_poor, pred_prob_good)
log(3)   # log loss of guessing 1/3 for every class, for comparison

# 9.5 Baseline and cross-validation
# baseline: accuracy of always predicting the largest class (poor)
data.frame(model    = c("Always predict poor", "Multinomial logistic regression"),
           accuracy = round(c(cm$overall["AccuracyNull"], cm$overall["Accuracy"]), 3),
           row.names = NULL)
cm$overall["AccuracyPValue"]   # one-sided test: accuracy > no-information rate

# 10-fold cross-validation on the training data, also tuning `decay`
# (the L2 / ridge penalty in multinom; decay = 0 is the unpenalized model above)
train_x <- as.data.frame(dplyr::select(train, -quality_group))
train_y <- factor(train$quality_group, levels = c("0", "1", "2"))

set.seed(123)   # reproducible fold assignment
cv_fit <- train(
  x = train_x, y = train_y,
  method     = "multinom",
  preProcess = c("center", "scale"),       # scaling redone inside each fold (no leakage)
  trControl  = trainControl(method = "cv", number = 10),
  tuneGrid   = expand.grid(decay = c(0, 0.001, 0.01, 0.1, 1)),
  trace      = FALSE                       # hide multinom's iteration log
)
round(cv_fit$results[, c("decay", "Accuracy", "Kappa")], 4)

png("img/fig-cv-decay.png", width = 600, height = 450)
print(plot(cv_fit))   # print() is required: lattice plots are not drawn automatically inside a script
dev.off()


# 10. Class imbalance

# good wines are rare, so the unweighted model rarely predicts "good".
# weight each training wine inversely to its class size, w = n / (3 * n_class),
# so all three classes contribute equally to the likelihood
counts <- table(train$quality_group)
round(nrow(train) / (3 * counts), 2)   # weight for each class

w <- as.numeric(nrow(train) / (3 * counts[train$quality_group]))   # one weight per wine

model_w <- multinom(quality_group ~ ., data = train, weights = w,
                    maxit = 1000, trace = FALSE)
model_w$convergence   # 0 = converged

# predictions and metrics for the weighted model (level order 0, 1, 2 as above)
pred_w <- factor(predict(model_w, newdata = test, type = "class"),
                 levels = c("0", "1", "2"))
cm_w <- confusionMatrix(pred_w, test_y, mode = "everything")
cm_w_counts <- table(Predicted = pred_w, Actual = test_y)
cm_w_counts

# values quoted in the vignette text
c(avg_called_good_unweighted = cm_counts["2", "1"],         # 11
  avg_called_good_weighted   = cm_w_counts["2", "1"],       # 33
  precision_good_unweighted  = round(cm$byClass["Class: 2", "Precision"], 2),    # 0.52
  precision_good_weighted    = round(cm_w$byClass["Class: 2", "Precision"], 2),  # 0.45
  wines_labeled_good         = sum(cm_w_counts["2", ]))     # 76

# recall by class: weighting should raise recall for good wines
data.frame(class             = c("poor", "average", "good"),
           recall_unweighted = round(cm$byClass[, "Recall"], 3),
           recall_weighted   = round(cm_w$byClass[, "Recall"], 3),
           row.names = NULL)

# overall accuracy and macro F1: weighting usually costs a little accuracy
data.frame(model    = c("unweighted", "weighted"),
           accuracy = round(c(cm$overall["Accuracy"], cm_w$overall["Accuracy"]), 3),
           macro_f1 = round(c(mean(cm$byClass[, "F1"]), mean(cm_w$byClass[, "F1"])), 3))


# 11. Multicollinearity

# correlations between predictors in the training set
# (standardizing does not change correlations)
cor_matrix <- cor(train[predictors])

# the five most strongly correlated pairs (each pair listed once)
as.data.frame(as.table(cor_matrix)) %>%
  filter(as.character(Var1) < as.character(Var2)) %>%
  mutate(abs_cor = abs(Freq)) %>%
  arrange(desc(abs_cor)) %>%
  head(5)

# correlation heatmap: blue = positive, red = negative
cor_plot <- as.data.frame(as.table(cor_matrix)) %>%
  ggplot(aes(x = Var1, y = Var2, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = round(Freq, 2)), size = 2.5) +
  scale_fill_gradient2(low = "firebrick", mid = "white", high = "steelblue",
                       limits = c(-1, 1)) +
  labs(title = "Correlation between predictors", x = NULL, y = NULL, fill = "r") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
cor_plot
ggsave("img/fig-correlation.png", cor_plot, width = 7, height = 6, dpi = 150)

# effect of correlated predictors: refit without density, which is correlated
# with both alcohol and fixed acidity, and compare their coefficients and SEs
model_no_density <- multinom(quality_group ~ . - density, data = train,
                             maxit = 1000, trace = FALSE)

terms <- c("alcohol", "`fixed acidity`")   # multinom keeps backticks around names with spaces
data.frame(
  class        = rep(c("poor", "good"), times = 2),
  term         = rep(c("alcohol", "fixed acidity"), each = 2),
  coef_with    = round(as.vector(coef(model)[, terms]), 3),
  coef_without = round(as.vector(coef(model_no_density)[, terms]), 3),
  se_with      = round(as.vector(summary(model)$standard.errors[, terms]), 3),
  se_without   = round(as.vector(summary(model_no_density)$standard.errors[, terms]), 3)
)

# test accuracy with and without density: predictions barely change
pred_nd <- factor(predict(model_no_density, newdata = test, type = "class"),
                  levels = c("0", "1", "2"))
c(with_density    = mean(test_y_pred == test_y),
  without_density = mean(pred_nd == test_y))


# 12. Extension: solution to the reference-class exercise
# (the vignette poses this as an exercise and shows no results from it)

# refit with poor ("0") as the reference class
train_ref0 <- train
train_ref0$quality_group <- relevel(train_ref0$quality_group, ref = "0")
model_ref0 <- multinom(quality_group ~ ., data = train_ref0, maxit = 1000, trace = FALSE)

# 1. predicted classes on the test set do not change
pred_ref0 <- predict(model_ref0, newdata = test, type = "class")
all(as.character(pred_ref0) == as.character(predicted_classes))   # expect TRUE

# 2. RRR for alcohol in the good vs. poor equation
exp(coef(model_ref0)["2", "alcohol"])

# 3. the same number from the original model: the good-vs-poor log-odds is the
#    difference of the two log-odds against average, because the average terms cancel
exp(coef(model)["2", "alcohol"] - coef(model)["0", "alcohol"])


# 13. Session info

# documents package versions for reproducibility
sessionInfo()