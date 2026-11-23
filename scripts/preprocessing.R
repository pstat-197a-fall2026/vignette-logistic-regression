library(dplyr)
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


# save preprocessed data into data folder
write.csv(wine, file = "data/preprocessed_winequality.csv", row.names = FALSE)

