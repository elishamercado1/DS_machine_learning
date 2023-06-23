library(mlr3verse)
library(data.table)
library("mlr3viz")
library("rpart.plot")

# CLASSIFICATION TASK -----------------------------------------------------

# Create a new classification task from iris
task_iris = TaskClassif$new(id = "iris", backend = iris, target = "Species")

# id: An identifier (label) for the task, used in plots and summaries. 
# backend: Used for specifying data objects. data.frame() objects are recommended for performance and compatibility. 
# target: Label of the target column.

# Print the task
print(task_iris)

# REGRESSION TASK ---------------------------------------------------------

# Load and prepare the data
data("mtcars", package = "datasets")
data_subset = mtcars[, 1:3]
str(data_subset)

# Create a regrssion task with "data" 
task_cars_subset = TaskRegr$new(id = "cars", backend = data_subset, target = "mpg")
print(task_cars_subset)

# BUILT IN TASKS ----------------------------------------------------------
# Retrieve all the ML tasks available in`mlr3`as a data table
as.data.table(mlr_tasks)

# load breast_cancer task from mlr_tasks
task_cancer <- mlr_tasks$get("breast_cancer")
print(task_cancer)


# SUGAR FUNCTIONS ---------------------------------------------------------
#Functions to retrieve objects, set hyperparameters and assign to fields in one go

# create a task using tsk()
task_housing <- tsk("boston_housing")
print(task_housing)

# VISUALISING TASKS -------------------------------------------------------

# visulize task_cancer
autoplot(task_cancer)

# visulize cars dataset as pair plots
autoplot(task_cars_subset, type = "pairs")


# TASK API ----------------------------------------------------------------
# use task API - TASK'S PUBLIC FIELDS AND METHODS
# get no of rows
task_cancer$nrow

# get no of cols
task_cancer$ncol

# get data
task_cancer$data()


# LEARNERS IN MLR3 --------------------------------------------------------

# Access the keys for mlr_leaners dictionaries
mlr_learners

# Get linear regression as implemented by stats::lm 
learner_regression <- mlr_learners$get("regr.lm")
print(learner_regression)

# or use the sugar function lrn() to get same results
learner_regression <- lrn("regr.lm")
print(learner_regression)

# Learners typically need the following:
# Parameters are hyperparameter values which can be set as needed.
# 
# Packages from R required for training and prediction stages.
# 
# Predict Type of learner e.g. predicting labels vs. probabilities etc.
# 
# Feature Types are type of features e.g. numeric, logical or discrete etc. that the learner can train on.
# 
# Properties of learners e.g. ability to handle missing values, evaluate feature importance w.r.t target variable etc.

# Access learner hyperparameters
learner_regression$param_set

# Assign new hyperparameter values to the learner to return x and y values after the fitting
learner_regression$param_set$values = list(x = TRUE, y = TRUE)
print(learner_regression)

learner_regression$param_set$values


# Model training ----------------------------------------------------------

# Uncomment and run the command below to create the task if you haven't done so already
# task_iris = TaskClassif$new("iris", iris, "Species")

## STEP 1: CREATE/RETRIEVE THE MACHINE LEARNING TASK
# Print the iris task and visualize it with pairplots
print(task_iris)
autoplot(task_iris, type = "pairs")

## STEP 2: CREATE THE LEARNER OBJECT
# Create a decision tree learner from classif.rpart
learner_classification <- mlr_learners$get("classif.rpart")

# Alternatively, using sugar function
#learner_classification <- lrn("classif.rpart")

print(learner_classification)

## STEP 3: SPLITTING THE DATA
# Create Train and Test sets
train_set = sample(task_iris$nrow, 0.8 * task_iris$nrow)
test_set = setdiff(seq_len(task_iris$nrow), train_set)

## STEP 4: TRAINING THE LEARNER
# Access the model field of the learner
learner_classification$model

# Train the leaner with task_iris
learner_classification$train(task_iris, row_ids = train_set)

# Print the trained model
print(learner_classification$model)

# Visualize the trained tree model 
rpart.plot(learner_classification$model)

# Show rules
rpart.rules(learner_classification$model)
# Also gives matrix with percentage of correctly and incorrectly identified observations

# Show hyperparameters
learner_classification$param_set

# MAKING PREDICTIONS ------------------------------------------------------
# Make predictions with test data  
predictions_iris <- learner_classification$predict(task_iris, row_ids = test_set)
print(predictions_iris)

# PERFORMANCE EVALUATION --------------------------------------------------
# Plot predictions vs. ground truth
autoplot(predictions_iris)

# Plot the confusion matrix 
predictions_iris$confusion

## EVALUATION WITH THE MEASURE OBJECT
# Retrieve keys from mlr_measure dictionary
as.data.table(mlr_measures)
# classification Accuracy measure which gives a probability of “correctly” 
# predicting an outcome through dividing the number correct predictions by total predictions count. 

# Instantiate classif.acc as the evaluation measure
accuracy_measure <- mlr_measures$get("classif.acc")
print(accuracy_measure)

# Or use the msr() sugar function
accuracy_measure = msr("classif.acc")
print(accuracy_measure)

# Pass in the measure to the score field of prediction object
predictions_iris$score(accuracy_measure)

# CHANGING HYPERPARAMETERS ------------------------------------------------
# Create a copy of the classification learner
learner_classification_copy <- learner_classification$clone(deep = TRUE)
# Set maxdepth hyperparameter to 1
learner_classification_copy $param_set$values = list(maxdepth=1)
print(learner_classification_copy)

# Retrain with new hyperparameter data
learner_classification_copy$train(task_iris, row_ids = train_set)
print(learner_classification_copy$model)

# Visualize the new model
rpart.plot(learner_classification_copy$model)

# Extract rules
rpart.rules(learner_classification_copy$model)
# worse than original model

# RESAMPLING IN MLR3 ------------------------------------------------------
# Resampling strategies are used to assess the performance of a learning algorithm
# Taking just one train/test split means our model is biased to predict based on 
# the single training set we gave it

## CROSS VALIDATION
# Divide training data into K equally sized partitions ("folds")
# train on each fold, test on others - get score and metric for K different models

## RESAMPLING WITH IRIS
# Create the task and decision tree learner for iris dataset
task = tsk("iris")
learner = lrn("classif.rpart")

# Access the mlr_resamplings dictionary
as.data.table(mlr_resamplings)

## HOLDOUT SAMPLING
# Simple 80/20 split is referred to as "holdout sampling"
# Argument ratio determines the ratio of observation going into the training set (default: 2/3)

# Create a holdout resampling object with an 80/20 split 
ho_resample <- mlr_resamplings$get("holdout")
ho_resample

# Or use rsmp() sugar function
ho_resample <- rsmp("holdout")
print(ho_resample)
# instantiated=FALSE means it hasn't yet been applied to a task

# Change the split level to 80/20
ho_resample$param_set$values = list(ratio = 0.8)
print(ho_resample)

# Initiate the resampling object - splits into test and train according to ratio
ho_resample$instantiate(task)
print(ho_resample)

# get the train and test data indices
train_idx <- ho_resample$train_set(1)
test_idx <- ho_resample$test_set(1)

# Print train and test indices
print(train_idx)

print(test_idx)

# Clone the task and filter rows using train and test index values
task_train <- task$clone()$filter(train_idx)
task_test <- task$clone()$filter(test_idx)
print(task_train)

print(task_test)

## RESAMPLING WITH CROSS VALIDATION
# Create the task and decision tree learner for iris dataset
task = tsk("iris")
learner = lrn("classif.rpart")

# use resample() function for resampling
crossval <- rsmp("cv", folds = 7L)
# store_model = TRUE allows us to examin individual results later
resamp_results <- resample(task, learner, crossval, store_models = TRUE)

print(resamp_results)

# Get aggregate results from cross-validation sampling
resamp_results$aggregate(msr("classif.acc"))

# Assess performance of individual folds
resamp_results$score(msr("classif.acc"))

# Can alse retrieve from specific iterations : Get the learner from iteration 2 of the results object
lrn = resamp_results$learners[[2]]
lrn$model
