# Saratoga house prices dataset
options(warn = -1) # supress version warnings

library(mlr3verse) # mlr3 complete ecosystem 
library(tidyverse, warn.conflicts=FALSE) # data cleaning
library(skimr) # summary statstics
library(DataExplorer) # Automation for EDA
library(ggpubr) # easy ggplot2
library(univariateML) # maximum liklihood estimation - MLE
library(mosaicData) # datasets 
library(DT) # datatable object for tabular data
library(data.table) # display the dictionary elements of mlr3
library(scales) # formatting nubers in ggplot axes text 

# DATA --------------------------------------------------------------------

# Load the Saratoga housing dataset from Mosaic data 
data("SaratogaHouses", package = "mosaicData")
saratoga_data <- SaratogaHouses

# Show the contents of data , use datatable() for pretty printing
datatable(saratoga_data, rownames = FALSE, filter="top", options = list(pageLength = 5, scrollX=T))

# EXPLORATORY DATA ANALYSIS -----------------------------------------------
# Inspect the dataset contents and statistical summary using skim()
skimr::skim(saratoga_data)

# Response/Target variable
# Plot the distribution+rug of the response variable "price". 

price_plot <- ggplot(data = saratoga_data, aes(x = price)) +
  
  geom_histogram(aes(y=..density..), colour="black", fill="gray75", bins=30) +        
  geom_density(fill = "skyblue4", alpha = 0.6, color= "slateblue") +
  geom_rug(colour = "skyblue4", alpha = 0.1) +
  
  scale_x_continuous(labels = scales::dollar_format()) +
  labs(title = "Distribution of Response Variable \"Price\" ") +
  theme_minimal() 
price_plot

# is skewed, try plotting log of price instead....
# Plot the distribution+rug of the response variable x`log(price). 

price_plot <- ggplot(data = saratoga_data, aes(x = log(price))) +
  
  geom_histogram(aes(y=..density..), colour="black", fill="gray75", bins=30) +        
  geom_density(fill = "skyblue4", alpha = 0.6, color= "slateblue") +
  
  geom_rug(colour = "skyblue4", alpha = 0.1) +
  labs(title = "Distribution of log(Price) ") +
  theme_minimal() 
price_plot

# Fitting distribution
# fit a normal distribution to data
mlnorm(saratoga_data$price)

# GOODNESS OF FIT WITH AIC
# calculate AIC
AIC(mlnorm(saratoga_data$price))

# compare AIC by fitting key distributions to the data
compare_aic <- AIC(
  mlnorm(saratoga_data$price),
  mlbetapr(saratoga_data$price),
  mlexp(saratoga_data$price),
  mlinvgamma(saratoga_data$price),
  mlgamma(saratoga_data$price),
  mllnorm(saratoga_data$price),
  mlrayleigh(saratoga_data$price),
  mlinvgauss(saratoga_data$price),
  mlweibull(saratoga_data$price),
  mlinvweibull(saratoga_data$price),
  mllgamma(saratoga_data$price)
)

# Arrange by increasing AIC
compare_aic %>% 
  rownames_to_column(var = "Distribution") %>% 
  arrange(AIC)
# Lower AIC is better

# FEATURE VARIABLES
#plot densities for all continuous features in the dataset
plot_density(
  data = saratoga_data %>% select(-price), # remove the price column
  ncol = 3,
  title = "Continuous Variables' Distributions",
  ggtheme = theme_minimal(),
  theme_config = list(
    plot.title = element_text(size = 16, face = "bold"),
    strip.text = element_text(colour = "black", size = 12, face = 2)
  ))

# Check the values in "fireplaces"
table(saratoga_data$fireplaces)

# change fireplace variable to factor type
saratoga_data <- saratoga_data %>% 
  mutate(fireplaces = as.factor(fireplaces))
head(saratoga_data, 1)

# Plotting correlations
# function to check correlation 
# inputs: 
#   var1: continuous variables list
#   var2: target variable
#   ds:   dataset

corr <- function(var1, var2, df, alpha=0.3){
  p <- df %>%
    mutate(
      # Give a suitable title for each plot
      title = paste(toupper(var1), "vs", toupper(var2))) %>%
    ggplot(aes(x = !!sym(var1), y = !!sym(var2))) + 
    geom_point(alpha = alpha) +
    # Check for linear relationships
    geom_smooth(se = FALSE, method = "lm", color = "firebrick") +
    # Check for non-linear relationships
    geom_smooth(se = FALSE, method = "gam", formula = y ~ splines::bs(x, 3)) +
    
    facet_grid(. ~ title) +
    theme_minimal() +
    theme(strip.text = element_text(colour = "black", size = 8, face = 2),
          axis.title = element_blank())
  return(p + scale_x_continuous(labels = comma))
}

# Identify the correlation between continuous vars and target variable 

continuous_vars <- c("livingArea", "pctCollege", "bedrooms",
                     "bathrooms", "rooms", "age", "lotSize",
                     "landValue")

# Map variables to the function
plots <- map(
  .x = continuous_vars,
  .f = corr,
  var2 = "price",
  df = saratoga_data
)

# display output in subplots`
ggarrange(plotlist = plots, ncol = 3, nrow = 3) %>%
  annotate_figure(
    top = text_grob("Correlation with price", face = "bold", size = 16, x = 0.20)
  )

# Log transformation
# apply log trasformation to skewed features 

saratoga_data <- saratoga_data %>%
  mutate(
    log_age = log10(age + 0.1),
    log_lotSize = log10(lotSize),
    log_landValue = log10(landValue)
  )

# Plot the log transformed feature with correlation() function

continuous_vars <- c( "log_age", "log_lotSize", "log_landValue")
plots <- map(
  .x = continuous_vars,
  .f = corr,
  var2 = "price",
  df = saratoga_data
)

ggarrange(plotlist = plots) %>%
  annotate_figure(
    top = text_grob("Correlation with price", face = "bold", size = 16,
                    x = 0.20)
  )

# Correlation heatmap
# remove log transformations
saratoga_data <- saratoga_data %>%
  select(-c(log_age, log_lotSize, log_landValue))

# plot correlation between features - from DataExplorer package
plot_correlation(
  data = saratoga_data,
  type = "continuous",
  title = "Continuous Variables Correlation Heatmap",
  theme_config = list(legend.position = "none",
                      plot.title = element_text(size = 12, face = "bold"),
                      axis.title = element_blank(),
                      axis.text.x = element_text(angle = -45, hjust = +0.1)
  )
)
# Multicollinearity between rooms and bathrooms and living space
# Don't remove because this makes sense and should still provide slightly different information

# CATEGORICAL FEATURES
# Plot bar charts to show class distribution for categorical variables

plot_bar(
  saratoga_data,
  ncol = 3,
  title = "Frequency of Observations per class",
  ggtheme = theme_minimal(),
  theme_config = list(
    plot.title = element_text(size = 16, face = "bold"),
    strip.text = element_text(colour = "black", size = 10, face = 2),
    legend.position = "none"
  )
)

# There is an imbalance in the number in the fireplace class...
table(saratoga_data$fireplaces)

# Recode to 2plus category...
saratoga_data <- saratoga_data %>%
  mutate(
    fireplaces = recode_factor(
      fireplaces,
      `2` = "2plus",
      `3` = "2plus",
      `4` = "2plus"))

table(saratoga_data$fireplaces)

# Custom box plot function with integrated violin plots 
custom_box_plot <- function(var1, var2, df, alpha=0.3){
  p <- df %>%
    mutate(
      
      # Set the title
      title = paste(toupper(var2), "vs", toupper(var1))
    ) %>%
    ggplot(aes( x = !!sym(var1), y = !!sym(var2))) + 
    
    # draw the violin plot
    geom_violin(alpha = alpha) +
    
    # draw the boxplot   
    geom_boxplot(width = 0.1, outlier.shape = NA) +
    
    facet_grid(. ~ title) +
    theme_minimal() +
    theme(strip.text = element_text(colour = "black", size = 8, face = 2),
          axis.text.x = element_text(size = 7),
          axis.title = element_blank())
  return(p + scale_y_continuous(labels = comma))
}

# list categorical variables
categorical_vars <- c("fireplaces", "heating", "fuel", "sewer","waterfront", "newConstruction", "centralAir")

# map the vars to function
plots <- map(
  .x = categorical_vars,
  .f = custom_box_plot,
  var2 = "price",
  df = saratoga_data
)

ggarrange(plotlist = plots, ncol = 3, nrow = 3) %>%
  annotate_figure(
    top = text_grob("Correlation with price", face = "bold", size = 16,
                    x = 0.16)
  )


# DATA MODELLING ----------------------------------------------------------

# Create a new REGRESSION task from saratoga data
task = TaskRegr$new(id = "task_saratoga", backend = saratoga_data, target = "price")
print(task)

autoplot(task)

# Check column information in the task 
task$col_info

# SPLITTING DATA
set.seed(1208)
saratoga_resample <- rsmp("holdout", ratio = 0.8)
print(saratoga_resample)

saratoga_resample$instantiate(task)
print(saratoga_resample)

# get the train and test data indices
train_idx <- saratoga_resample$train_set(1)
test_idx <- saratoga_resample$test_set(1)

# Clone the task and filter rows using train and test index values
task_train <- task$clone()$filter(train_idx)
task_test <- task$clone()$filter(test_idx)
print(task_train)

print(task_test)

# verify that distribution of the response variable is similar in training and test set
summary(task_train$data()[["price"]]); summary(task_test$data()[["price"]])
# Difference in mins because of outliers at low end


# PREPROCESSING DATA WITH MLR3 PIPELINES ----------------------------------

# Get a list of pipeops

mlr_pipeops

# Get a list of filters
mlr_filters

# create single pipeops
pca = mlr_pipeops$get("pca")

# or do so using sugar function:
pca = po("pca")

# PREPROCESSING PIPEOPS
# create a graph object with scale and pca pipeops
graph = mlr_pipeops$get("scale") %>>% mlr_pipeops$get("pca")

# Create a preprocessing pipeline with scale and encode pipeops

preprocessing_pipeline <- po("scale", param_vals = list(center = TRUE, scale = TRUE)) %>>%
  po("encode", param_vals = list(method = "one-hot")
  )
# plot the pipeline
preprocessing_pipeline$plot()

# FITTING PIPELINE TO DATA
# Fit the pipeline to the training task 
preprocessing_pipeline$train(task_train)
# Output is always a list in which the first element ([[1]]) contains the transformed data

# Can save preprocessed test and train data as new tasks by using the $predict$ method
# Create the new pre-procesed test and train tasks from proprocesing pipeline. 
task_train_prep <- preprocessing_pipeline$predict(task_train)[[1]]$clone()
task_test_prep <- preprocessing_pipeline$predict(task_test)[[1]]$clone()
task_train_prep$data() %>% head()


# MODEL DEVELOPMENT -------------------------------------------------------
# get the keys of all mlr learners
mlr_learners$keys()

# MULTIPLE LINEAR REGRESSION WITH REGR.LM
learner_lm <- mlr_learners$get("regr.lm")
print(learner_lm)

# MODEL TRAINING
learner_lm$train(task = task_train_prep)
print(learner_lm$model)


# MAKING PREDICTIONS ------------------------------------------------------
preds <- learner_lm$predict(task = task_test_prep) 
as.data.table(preds) |> head() 

# REGRESSION ERROR --------------------------------------------------------

# REGRESSION ERROR IN MLR3
print(mlr_measures)

# Use rmse here because is directly interpretable in terms of our measurement units (price)
# Calculate the rmse from predictions object
preds$score(measures = msr("regr.rmse"))
###THIS DOESN@T WORK BECAUSE SOME VALUES ARE INF ###

autoplot(preds)

# CALCULATING RESIDUALS
# Calculate the residuals for predictions. 
pred_validation <- as.data.table(preds) %>%
  mutate(
    res = response - truth # residulas
  )
print(pred_validation)

# VISUALISING RESIDUALS
# Plot predicted vs actual values with a regression line
plot1 <- ggplot(data = pred_validation, aes(x = truth, y = response)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, color = "firebrick") +
  labs(title = "Predicted value vs actual value") +
  theme_minimal()

# Plot loss in each prediction 
plot2 <- ggplot(data = pred_validation, aes(x = row_ids, y = res)) +
  geom_point(alpha = 0.3) +
  geom_hline(yintercept = 0, color = "firebrick") +
  labs(title = "Model residuals") +
  theme_minimal()

# Plot the distribution of residuals
plot3 <- ggplot(data = pred_validation, aes(x = res)) +
  geom_density() + 
  labs(title = "Residual distribution of the model") +
  theme_minimal()

# plot qq plot to compare against normal distribution
plot4 <- ggplot(data = pred_validation, aes(sample = res)) +
  geom_qq() +
  geom_qq_line(color = "firebrick") +
  labs(title = "Q-Q residuals of the model") +
  theme_minimal()

ggarrange(plotlist = list(plot1, plot2, plot3, plot4)) %>%
  annotate_figure(
    top = text_grob("Residuals Distribution", size = 15, face = "bold")
  )

