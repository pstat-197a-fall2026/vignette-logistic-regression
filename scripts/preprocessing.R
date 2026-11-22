library(dyplr)

library(readr)
wine <- read_csv("data/winequality-red.csv")
summary(wine)

# change numerical wine quality scores into our three categorical values
wine$quality <- ifelse(wine$quality < 4, 0,
                       ifelse(wine$quality <= 6, 1, 2))
summary(wine$quality)
