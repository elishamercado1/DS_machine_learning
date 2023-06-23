# Chapter 4 classification analysis -------------------------------------------------


# 1. Dataset --------------------------------------------------------------

library(mlr3)
library(mlr3verse)
library(mlr3data)
library(DT)
library(skimr)
library(tidyverse)
library(precrec)
library(ranger)

set.seed(851) # random seed


titanic_dt <- datatable(titanic)

titanic_dt

skim(titanic)


# Load and prepare the data - split into test and train (first 891 observations)
data("titanic", package = "mlr3data")
train_data = titanic[1:891, ]
test_data = titanic[892:1309, ]

# Explore train_data
skim(train_data)
# No missing values for survived any more, a couple for age and embarkation point
# These will be dealt with later

# Class distribution of target variable (survived) is unbalanced: 549 no vs 342 yes
# Some variables eg. name, cabin, ticket don't carry any discriminating info about survival - consider removing later

# 2. Class distributions -----------------------------------------------------

# Classes with higher count will be predicted with better accuracy
# Often, classifiers will be modelled around negative examples to predict the worst outcome with the best accuracy
# For titanic dataset this is fine
# For eg. healthcare data where we mind want to predict whether a patient has a disease or not
# most patients will be healthy so we cannot use this data to train as is!
# To deal with it, we usually will want to oversample the minority class


# 2.1 SMOTE --------------------------------------------------------------
# https://arxiv.org/abs/1106.1813


# 3. The First Model - Decision Trees -------------------------------------
# EXERCISE:
# 1. Create a classification task with "data" 
titanic_task = TaskClassif$new(id = "titanic", backend = train_data, target = "survived")
print(titanic_task)

# 2. Remove columns with string values
titanic_task$select(cols = setdiff(titanic_task$feature_names, c("cabin", "name", "ticket")))
titanic_task

# 3. Create cross validation resampling instance with 7 folds and initiate with classification task
# use resample() function for resampling
set.seed(1208)
titanic_cv <- rsmp("cv", folds = 7L)
titanic_cv$instantiate(titanic_task)
print(titanic_cv)

# 4. Set up decision tree learner object classif.rpart
learner = lrn("classif.rpart")
print(learner)

# 5. Use resample() function with task, learner, resampling object
resamp_titanic <- resample(titanic_task, learner, titanic_cv, store_models = TRUE)

# 6. Calculate the aggregate performance of cross validation with classif.accuracy from the output of resmaple()
# Get aggregate results from cross-validation sampling
resamp_titanic$aggregate(msr("classif.acc"))
# 0.8102942
# >80% accuracy with just our simple model

# Assess performance of individual folds
resamp_titanic$score(msr("classif.acc"))

# 4. LOGISTIC REGRESSION --------------------------------------------------
# Logistic function is fitted to the data. Value of this function corresponds to the
# probability that a data point is in a certain class using Maximum Likelihood Estimation.
# Mathematically the logistic function looks like: f(x) = 1/(1+e^-x)

# 4.1 Logistic regression in mlr3

learner_logreg <- mlr_learners$get("classif.log_reg") # get from learner directory
learner_logreg$param_set

# Imputing missing values so hopefully regression will run
train_clean <- train_data |> 
  select(-cabin, -name, -ticket) |> 
  mutate(age = ifelse(is.na(age), mean(age, na.rm = TRUE), age), 
         embarked = ifelse(is.na(embarked), mode(embarked), embarked))

# Creating a new classification task with imputed training data
titanic_task = TaskClassif$new(id = "titanic", backend = train_clean, target = "survived")
print(titanic_task)

# Train the model and inspect model diagnostics
learner_logreg$train(task = titanic_task)
summary(learner_logreg$model)

# Train a logistic regression learner with corss val. 
res <- resample(titanic_task, learner_logreg, titanic_cv, store_models = TRUE)
agg <- res$aggregate(msr("classif.acc"))
agg


# 5. MODEL EVALUATION -----------------------------------------------------
# 5.1 Truth tables
# Here assign "no" = "positive" prediction, "yes" = "negative"
# True +ve = correct prediction of "no"
# False +ve = incorrect prediction of "no" (predicts "yes" for a "no")
# True negative = correct "yes" prediction
# False negative = incorrect "yes" prediction (predicts "no" for a "yes")

# 5.2 Confusion matrix
# Allows to visualize how model performs with each target class
# so can understand if it is favouring certain classes disproportionately
# y-axis is the TRUE value of the target, x is PREDICTED value

# Calculate train and test indices by random sampling the data
n = titanic_task$nrow
train_idx = sample(seq_len(n), size = round(n * 0.8))
test_idx = setdiff(seq_len(n), train_idx)

# train the model on training indices
learner_logreg$train(task = titanic_task, row_ids = train_idx)
# generate the prediction
pred = learner_logreg$predict(task = titanic_task, row_ids = test_idx)

# View the head of pred object with fortify() 
head(fortify(pred))

# Print the confusion matrix
print (pred$confusion)

# Plot predictions and true labels
autoplot(pred)
# Model is more successful at predicting "no" than "yes"

# 5.3 Recall
# Recall = true positive rate
# recall = no. true +ve predictions / no. true positives = #TP/(#TP +#FN)

# 5.4 Precision
# Proportion of true positives compared with the total predicted positive results
# precision = no. true positive predictions / no. predicted positive = #TP/(#TP+#FP)

# 5.5 F1 Score
# 5.5.1 Healthcare example
# In situations such as healthcare, we would prefer to have a false positive compared
# with missing a true positive (predict someone has a disease when they don't vs 
# not picking up a diagnosis). Therefore we want to maximise recall and we don't care so much
# about precision.
# Precision and recall metrics can be combined to get the F1 score which tells us how well
# a model performs wrt all the true +ve values and all the predicted true values
# F1 = (2*precision*recall)/(precision+recall) using the true positive notation:
# F1 = TP/(TP + 0.5FP + 0.5FN)

# We can calculate all these using mlr_measures as shown below:
msrs = lapply(c("classif.precision", "classif.recall", "classif.fbeta"), msr)
pred$score(measures = msrs)

# 5.6 Receiver operating characteristic (ROC)
# ROC curve = a plot which shows how true +ve rate varies with false +ve over a range of thresholds
# Will be completely diagonal (TP rate = FP rate) if model is no better than random chance
# The further above diagonal it is, the better the model is performing

# 5.7 Precision recall curve
# Plot of precision (y-axis) and recall (x-axis) for different probability thresholds
# recommended for highly skewed examples where ROC curves might provide an excessively optimistic view of performance

# 5.8 Area under curve
# = area under ROC curve (0-1). AUC of 0.5 is equivalent to the same prediction power as random
# The closer the value is to 1 the better the model

# To calculate AUC or ROC we need probabilities of membership to each class
# instead of class labels themselves:
learner_logreg_prob = lrn("classif.log_reg", predict_type = "prob")

# train the model on training indices
learner_logreg_prob$train(task = titanic_task, row_ids = train_idx)
# generate the prediction
pred = learner_logreg_prob$predict(task = titanic_task, row_ids = test_idx)
print(fortify(pred))

# Plot ROC curve
autoplot(pred, type = "roc")

# Plot RPC curve
autoplot(pred, type = "prc")

# Plot AUC curve
pred$score(msr("classif.auc"))


# 6. DATA PREPROCESSING ---------------------------------------------------

# Original data task
titanic_task = TaskClassif$new(id = "titanic", backend = train_data, target = "survived")
titanic_task$select(cols = setdiff(titanic_task$feature_names, c("cabin", "name", "ticket")))

# check for missing values
titanic_task$missings()

# get a list of imputation pipeops
impute_pos <- as.data.table(mlr_pipeops)[grepl("impute", key)] # show available pipeops
impute_pos

# 6.1 Data imputation with pipeops

# Set up pre-processing strategy for numbers and factors
impute_nums = c("imputehist", "imputemedian", "imputemean") # imputing strategies for numerical values
impute_fcts = c( "imputesample", "imputemode") #  categorical (factor) values

# Introduce branching to be able to process both numerical and factor variables separately
# Branching pipeops for numerical and factor imputation
po_branch_nums = po("branch", options = impute_nums, id = "brnch_nums")
po_branch_fcts = po("branch", options = impute_fcts, id = "brnch_fcts")

# Pipeops for numerical imputation
pos_impute_nums = lapply(impute_nums, po)
pos_impute_nums = gunion(pos_impute_nums) #disjoint union of graphs - for graph learners

# Pipeops for factor imputation
pos_impute_fcts = lapply(impute_fcts, po)
pos_impute_fcts = gunion(pos_impute_fcts)

# Build complete pipe
pipe = po_branch_nums %>>% 
  pos_impute_nums %>>% 
  po("unbranch", id = "unbr_nums") %>>% 
  po_branch_fcts %>>% 
  pos_impute_fcts %>>% po("unbranch", id = "unbr_fcts") %>>% 
  po("learner", learner = lrn("classif.log_reg"))

# Plot the pipes layout.
plot(pipe)

# Create Graphlearner object
graph_learner = GraphLearner$new(pipe)

# We can now use resample() with graph learner to train the whole graph created above.
# Train the graph learner created above on titanic_task using resampling strategy created earlier

resampler <- rsmp("cv", folds = 7)
resampling_results <- resample(
  task = titanic_task,
  learner = graph_learner,
  resampling = resampler,
  store_models = TRUE
)

# Calculate validation score
# Validation Loss
resampling_results$aggregate(measures = msr("classif.acc"))

# 7. RANDOM FOREST --------------------------------------------------------
# Build an ensemble of decision trees trained with the "bagging" method
# Bagging uses a combination of learning models to improve the overall result
# mlr3 random forest classifier = classif.ranger
# Exercise

# 1. Using the preprocessing pipeline built earlier, connect a “classif.ranger” with probabilities prediction
# Build complete pipe
ranger_pipe = po_branch_nums %>>% 
  pos_impute_nums %>>% 
  po("unbranch", id = "unbr_nums") %>>% 
  po_branch_fcts %>>% 
  pos_impute_fcts %>>% po("unbranch", id = "unbr_fcts") %>>% 
  po("learner", learner = lrn("classif.ranger"), predict_type = "prob")

# 2. Instantiate a graph learner using the pipeline
forest_graph_learner = GraphLearner$new(ranger_pipe)

# 3. train the graph learner with imputed data and 7 fold cross validation

resampler <- rsmp("cv", folds = 7)
resampling_results <- resample(
  task = titanic_task,
  learner = forest_graph_learner,
  resampling = resampler,
  store_models = TRUE
)

# 4. Calculate evaluation metrics
resampling_results$aggregate(measures = lapply(c("classif.acc", 
                                                 "classif.precision", 
                                                 "classif.recall", 
                                                 "classif.fbeta", 
                                                 "classif.auc"), msr))

# 5. Plot ROC and PRC
# Plot ROC curve
autoplot(pred, type = "roc")

# Plot RPC curve
autoplot(pred, type = "prc")

# Increase in classification accuracy. Also increases computational cost required by algorithm

# 7.1 Model explainability
# Explainable AI is artificial intelligence in which the results of the solution 
# can be understood by humans. It contrasts with the concept of the “black box” in 
# machine learning where even its designers cannot explain why an AI arrived at a 
# specific decision

# Random forest is mostly black box whereas decision trees and logistic regression are examplainable


# 8. BENCHMARKING ------------------------------------------------------------

# Exercise
# 1. Use the preprocessing pipeline (without the learner) to process titanic_task into titanic_task_prep (Use $train() and $predict()with pipeline)

pipe = po_branch_nums %>>% 
  pos_impute_nums %>>% 
  po("unbranch", id = "unbr_nums") %>>% 
  po_branch_fcts %>>% 
  pos_impute_fcts %>>% po("unbranch", id = "unbr_fcts") 

# Create a preprocessed task
# access train method of pipe
pipe$train(titanic_task)[[1]]

titanic_task_prep <- pipe$predict(titanic_task)[[1]]

# Create a list of learners for benchmarking 
lrnrs <- lapply(c("classif.log_reg", "classif.rpart", "classif.ranger"), lrn, predict_type = "prob" )
lrnrs

# 8.1 Benchmark design
# A “benchmark design” is essentially a matrix of settings you want to execute.
# Set up a resampling grid 
design = benchmark_grid(
  tasks = titanic_task_prep, 
  learners = lrnrs, 
  resamplings =  rsmp("cv", folds = 7)
)

# conduct benchmark using the benchmark() function
res2 = benchmark(design)

# See results of benchmarking based on classification accuracy
head(fortify(res2))

# plot benchmark results

# Plot boxplots
autoplot(res2)

#Plot ROC
autoplot(res2, type = "roc")

#Plot PRC
autoplot(res2, type = "prc")

# Calculate statistcial performance measures
measures = list(
  msr("classif.acc", id="acc"),
  msr("classif.auc", id ="auc"),
  msr("classif.precision", id="prec"),
  msr("classif.recall", id="rec"),
  msr("classif.fbeta", id="fscore"))
res2$aggregate(measures)

# Random forest generally performs better than decision tree or logistic regression
# in terms of overall accuracy
# Log reg is better than decisions trees for AUC and recall


# 9. FURTHER EXPERIMENTATION ----------------------------------------------


# We have used a preprocessing pipeline with a number of POs , with a motivation of explaining complex pipelines. Investigate the pipeline and other available POs to see if your imputation strategy can improve predictive performance.
# 
# Try running parameter tuning with random forest algorithm i.e. optimizing tree depth, max trees etc. Above, we ran the experiment with default values and you may see improvement with tuning.
# 
# Apply the techniques shown in the lesson to your own datasets.
