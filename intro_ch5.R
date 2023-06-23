
# 5. CLUSTER ANALYSIS -----------------------------------------------------


# 6. DATASET --------------------------------------------------------------

# Load necessary package
library(mlr3verse)
library(mlr3cluster)
library(dplyr)
library(GGally) # pairplots
library(DT)
library(mclust) # ARI calculation
library(factoextra) # dendrograms
library(datasets)
library(janitor)

data(iris)
datatable(iris, rownames = FALSE, filter="top", options = list(pageLength = 5, scrollX=T) )

# Use ggcatsmat() to create pairplots. 
GGally::ggscatmat(
  data = iris , color = "Species",alpha = 0.8) +
  theme_minimal() +
  labs(title = "Pairwise Correlation") +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    axis.text = element_blank(),
    strip.text = element_text(colour = "black", size = 5, face = 1)
  )

# Exercise
# 1. Create a features dataset from features and a truth vector from target species
iris_data <- iris |> 
  clean_names()

features <- iris_data |> 
  select(sepal_length, sepal_width, petal_length, petal_width)

truth <- iris_data |> 
  select(species)

# 2. Create new clustering task with features dataset
task_cluster = TaskClust$new(id = "clusters", backend = features)
task_cluster

# 6.1 Feature scaling
# Difficult to measure distance between data points if all variables are on different scales
