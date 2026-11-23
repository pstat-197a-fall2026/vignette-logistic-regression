library(dplyr)
library(ggplot2)
library(nnet)
library(caret)

library(readr)
wine <- read_csv("data/winequality-red.csv")
dim(wine)
head(wine)
str(wine)
summary(wine)

# change numerical wine quality scores into our three categorical values
# low quality: score <= 5
# medium quality: score = 6
# high quality: 7 <= score
wine$quality_group <- ifelse(wine$quality <= 5, 0,
                       ifelse(wine$quality <= 6, 1, 2))
wine$quality_group <- factor(wine$quality_group,
                             levels = c(0, 1, 2))
summary(wine$quality)
summary(wine$quality_group)

# remove original quality score
wine$quality <- NULL
str(wine)

# data points in each category/class
table(wine$quality_group)
prop.table(table(wine$quality_group))








set.seed(123)

# stratified sampling (bc of imbalanced classes)
train_indices <- unlist(
  lapply(levels(wine$quality_group), function(group) {
    
    group_indices <- which(wine$quality_group == group)
    
    sample(
      group_indices,
      size = length(group_indices) -
        max(1, floor(0.2 * length(group_indices)))
    )
  })
)

# training and testing datasets
train <- wine[train_indices, ]
test <- wine[-train_indices, ]

# class distributions of the training and test split
table(train$quality_group)
table(test$quality_group)




# set medium quality as the reference category
train$quality_group <- relevel(
  train$quality_group,
  ref = "1"
)

test$quality_group <- factor(
  test$quality_group,
  levels = c("1", "0", "2")
)


# multinomial logistic regression
model <- multinom(
  quality_group ~ .,
  data = train
)
summary(model)

exp(coef(model))
head(round(fitted(model), 2))




# predict most likely category for train dataset
predicted_classes <- predict(
  model,
  newdata = test,
  type = "class"
)


# predict probabilities for train dataset
predicted_probabilities <- predict(
  model,
  newdata = test,
  type = "probs"
)

head(predicted_probabilities)



# compare actual and predicted values
results <- data.frame(
  Actual = test$quality_group,
  Predicted = predictions
)

head(results)

confusion_matrix <- as.data.frame(table(Predicted = factor(predicted_classes, levels = c("0", "1", "2")),
                                                           Actual = factor(test$quality_group, levels = c("0", "1", "2"))))
confusion_matrix
confusion_matrix_plot <- ggplot(confusion_matrix, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq)) +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(
    title = "Multinomial Logistic Regression",
    subtitle = "Confusion Matrix",
    x = "Predicted Quality",
    y = "Actual Quality"
  )

confusion_matrix_plot

