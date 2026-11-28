# vignette-multinomial-logistic-regression

Vignette on multinomial logistic regression for predicting red wine quality from chemical measurements; created as a class project for PSTAT197A in Fall 2026.

## Contributors

- Kyle Costabile
- Shiv Dutta
- Alison Gu
- Caroline Liu
- Chris Zaragoza

## Vignette abstract

Multinomial logistic regression extends binary logistic regression to outcomes with more than two classes. This vignette shows how to fit and evaluate the model in R with `nnet::multinom()`, using the red wine quality data (1,599 wines, 11 chemical measurements) from Cortez et al. (2009). Quality scores are grouped into three classes: poor (score ≤ 5), average (6) and good (≥ 7). The vignette walks through a stratified train/test split, standardizing predictors, choosing a reference class, interpreting coefficients as relative risk ratios, and evaluating predictions with a confusion matrix, per-class precision, recall and F1, ROC curves, and log loss. On the test set the model reaches 63.5% accuracy, compared with 46.5% from always guessing the largest class. The rare "good" class is the hardest to predict, and the vignette shows how class weights raise its recall from 0.35 to 0.79. It also covers multicollinearity, cross-validation, and an ordinal-model extension.

## Repository contents

```
vignette-logistic-regression
|-- data
|   |-- winequality-red.csv            raw red wine data (Cortez et al., 2009)
|   |-- preprocessed_winequality.csv   data with the three-class outcome (created by the script)
|   |-- averaging-scores.csv           macro, weighted and micro metric averages (created by the script)
|-- img
|   |-- fig-*.png                      figures saved by the script
|-- scripts
|   |-- vignette-script.R              line-annotated script that reproduces every result in the vignette
|-- vignette.Rmd                       the vignette (R Markdown source)
|-- vignette.html                      the rendered vignette
|-- vignette-multiclass-logistic-regression.Rproj   RStudio project file
|-- README.md
```

- **`vignette.html`** is the main document. Open it in a browser to read the vignette.
- **`vignette.Rmd`** is the source for the vignette. Knitting it regenerates `vignette.html`.
- **`scripts/vignette-script.R`** runs the whole analysis from the raw data to every table and figure in the vignette, with a comment on each step. Its section numbers match the vignette's.

## Usage

1. Install R and RStudio, then install the required packages:

   ```r
   install.packages(c("ggplot2", "nnet", "caret", "tidymodels", "readr", "pROC", "rmarkdown"))
   ```

2. Open `vignette-multiclass-logistic-regression.Rproj` in RStudio. The scripts use relative paths, so R must be running from the repository root.

3. Reproduce the results by running the script:

   ```r
   source("scripts/vignette-script.R", echo = TRUE)
   ```

   `echo = TRUE` prints every result to the console. Figures are saved to `img/`.

4. To rebuild the vignette, open `vignette.Rmd` and click Knit.

The analysis was run with R 4.5.2. Results are reproducible because the train/test split and the cross-validation folds use a fixed seed.

## Contributing

To suggest a change, open an issue or fork the repository, make your changes on a branch, and open a pull request. If you change the analysis, re-run `scripts/vignette-script.R` and re-knit `vignette.Rmd` so the script, the vignette and the figures stay consistent.

## References

1. Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), 547–553.
2. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.), Section 4.3.5. Springer.
3. Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.), Chapter 8. Wiley.
4. Venables, W. N., & Ripley, B. D. (2002). *Modern Applied Statistics with S* (4th ed.). Springer. (Source of the `nnet` package.)
5. Kuhn, M. (2008). Building predictive models in R using the caret package. *Journal of Statistical Software*, 28(5), 1–26.
6. Robin, X., et al. (2011). pROC: an open-source package for R and S+ to analyze and compare ROC curves. *BMC Bioinformatics*, 12, 77.
7. Data: [UCI Machine Learning Repository, Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) and [Kaggle, Red Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009).