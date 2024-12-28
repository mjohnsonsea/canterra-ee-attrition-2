#### OPAN 6602 - Project 3 ####

# SAXA 3 #

# Mike Johnson | Kesh Kamani | Ryan Mathis | Khushi Patel | Andrew Singh #

### Set up ----

# Libraries
library(tidyverse)
library(caret)
library(GGally)
library(broom)
library(car) # Variance inflation factor
library(readxl) # read excel files
library(pROC) #Sampling-over and under, ROC and AUC curve
library(margins) # for marginal effects
library(rpart.plot)
library(randomForest) # needed for random forest models

# Set random seed for reproducibility
set.seed(206)

# Set viz theme
theme_set(theme_classic())

### Load Data ----
df = read_excel("data-raw/Employee_Data_Project.xlsx")

# Data structure
str(df)

# Update data types
df = 
  df %>% 
  mutate(
    # Dependent Variable
    Attrition = factor(Attrition),
    
    # Predictors
    BusinessTravel = factor(BusinessTravel),
    Education = factor(Education, levels = 1:5, labels = c("Below College", "College", "Bachelor", "Master", "Doctor")),
    Gender = factor(Gender), 
    JobLevel = factor(JobLevel),
    MaritalStatus = factor(MaritalStatus),
    NumCompaniesWorked = as.numeric(NumCompaniesWorked),
    TotalWorkingYears = as.numeric(TotalWorkingYears), 
    EnvironmentSatisfaction = factor(EnvironmentSatisfaction, levels = 1:4, labels = c("Low", "Medium", "High", "Very High")), 
    JobSatisfaction = factor(JobSatisfaction, levels = 1:4, labels = c("Low", "Medium", "High", "Very High"))) 

# Remove Irrelevant Columns
df = 
  df %>% 
  select(
    -EmployeeID,
    -StandardHours)

# Check for NA's
na_summary = df %>% 
  summarise_all(~ sum(is.na(.))) %>%
  pivot_longer(cols = everything(),
               names_to = "variable",
               values_to = "na_count") %>% 
  filter(na_count > 0)

# How should we handle NAs?
na_summary

# Drop NA values
df = na.omit(df)



### Step 1: Create a train/test split ----

# Divide 30% of data to test set
test_indices = createDataPartition(1:nrow(df),
                                   times = 1,
                                   p = 0.3)

# Create training set
df_train = df[-test_indices[[1]], ]

# Create test set
df_test = df[test_indices[[1]], ]


### Step 2: Data Exploration ----

# Summary of training set
summary(df_train)

#df_train %>% 
# ggpairs(aes(color = Attrition, alpha = 0.4))

# Viz of attrition distribution
# Imbalanced classes. Will need to downsample.
df_train %>% 
  ggplot(aes(x = Attrition)) + 
  geom_bar(fill = "steelblue") +
  labs(title = "Attrition Distribution")

# Viz of relationship between Education and Attrition
df_train %>% 
  ggplot(aes(x = Gender, fill = Attrition)) +
  geom_bar() +
  facet_grid(~Attrition) +
  labs(title = "Gender Distribution by Attrition") +
  scale_fill_manual(values = c("No" = "blue", "Yes" = "red"))


# Viz of relationship between Age and Attrition
df_train %>% 
  ggplot(aes(x = Age, fill = Attrition)) +
  geom_histogram(binwidth = 10, position = "dodge") +
  facet_grid(~Attrition) +
  labs(title = "Age Distribution by Attrition") +
  scale_fill_manual(values = c("No" = "blue", "Yes" = "red"))

df_train %>% 
  mutate(age_t = log(Age)) %>% 
  ggplot(aes(x = age_t, fill = Attrition)) +
  geom_histogram() +
  facet_grid(~Attrition) +
  labs(title = "Age Transformed Distribution by Attrition")
  

# Viz of relationship between Education and Attrition
df_train %>% 
  ggplot(aes(x = Education, fill = Attrition)) +
  geom_bar() +
  facet_grid(~Attrition) +
  labs(title = "Education Distribution by Attrition") +
  scale_fill_manual(values = c("No" = "blue", "Yes" = "red"))

# Viz of relationship between Job Satisfaction and Attrition
df_train %>% 
  ggplot(aes(x = JobSatisfaction, fill = Attrition)) +
  geom_bar() +
  facet_grid(~Attrition) +
  labs(title = "Job Satisfaction Distribution by Attrition") +
  scale_fill_manual(values = c("No" = "blue", "Yes" = "red"))

# Viz of relationship between Working Years and Attrition
df_train %>% 
  ggplot(aes(x = TotalWorkingYears, fill = Attrition)) +
  geom_histogram(binwidth = 5, position = "dodge") +
  facet_grid(~Attrition) +
  labs(title = "Working Years Distribution by Attrition") +
  scale_fill_manual(values = c("No" = "blue", "Yes" = "red"))

df_train %>% 
  mutate(workingyears_t = log(TotalWorkingYears)) %>% 
  ggplot(aes(x = workingyears_t, fill = Attrition)) +
  geom_histogram() +
  facet_grid(~Attrition) +
  labs(title = "Working Years Transformed Distribution by Attrition")

# Viz of distribution of age and marital status

df_train %>% 
  ggplot(aes(x = Age)) +
  geom_histogram(binwidth = 10, position = "dodge") +
  facet_grid(Attrition ~ MaritalStatus) +
  labs(title = "Age Distribution of Employees by Marital Status and Attrition")


df_train %>% 
  filter(Attrition == "Yes") %>% 
  ggplot(aes(x = Age)) +
  geom_histogram(binwidth = 10) +
  facet_grid(~MaritalStatus) +
  labs(title = "Age Distribution of Employees by Marital Status",
       subtitle = "Attritioned Employees") 

### Step 3: Data pre-processing ----

# Downsampling
downsample_df = downSample(x = df_train[ , colnames(df_train) != "Attrition"],
                           y = df_train$Attrition)

colnames(downsample_df)[ncol(downsample_df)] = "Attrition"

downsample_df %>% 
  ggplot(aes(x = Attrition)) + 
  geom_bar(fill = "steelblue") +
  labs(title = "Attrition Distribution")

### Step 4: Feature Engineering ----

### Logistic Regression ----

### Step 5: Feature & Model Selection ----

# Initial Model
f1 = glm(
  Attrition ~ .,
  data = downsample_df,
  family = binomial("logit"))

summary(f1)

vif(f1)

roc1 = roc(
  data = 
    tibble(
      actual = 
        df_train %>%  # not using balanced data for evaluation
        select(Attrition) %>% 
        unlist(),
      predicted = predict(f1, df_train)),
  "actual",
  "predicted"
)

plot(roc1)

roc1$auc

# Stepwise Regression
f_step = step(object = f1,
              direction = "both")

summary(f_step)

vif(f_step)

roc_step = roc(
  data = 
    tibble(
      actual = 
        df_train %>%  # not using balanced data for evaluation
        select(Attrition) %>% 
        unlist(),
      predicted = predict(f_step, df_train)),
  "actual",
  "predicted"
)

plot(roc_step)

roc_step$auc
roc1$auc

# Final Model
f_log = train(
  Attrition ~ 
    BusinessTravel +
    MaritalStatus +
    NumCompaniesWorked +
    TotalWorkingYears +
    TrainingTimesLastYear +
    YearsAtCompany +
    YearsWithCurrManager +
    EnvironmentSatisfaction +
    JobSatisfaction
    ,
  method = "glm",
  family = "binomial",
  data = downsample_df,
  trControl = trainControl(method = "cv",
                           number = 10,
                           classProbs = TRUE, # Enable probability predictions
                           summaryFunction = twoClassSummary), # Use twoClassSummary to compute AUC
  metric = "ROC" # ROC gives us AUC
)

# Final Model Results
summary(f_log)

f_log$finalModel


### Step 6: Model Validation ----

f_log$resample

f_log$results

p_log = predict(f_log, df_train, type = "prob")

roc_log = roc(df_train$Attrition, p_log$Yes)

plot(roc_log)

roc_log$auc

### Step 7: Predictions and Conclusions ----

p_log_test = predict(f_log, df_test, type = "prob")

roc_log_test = roc(df_test$Attrition, p_log_test$Yes)

plot(roc_log_test, main = "ROC Curve for Logistic Regression Model")

roc_log_test$auc


# Re-train the model on the whole data set for marginal effects/production

# Downsampling
downsample_prod = downSample(x = df[ , colnames(df) != "Attrition"],
                             y = df$Attrition)

colnames(downsample_prod)[ncol(downsample_prod)] = "Attrition"

# Production Model
f_log_prod = glm(
  Attrition ~ 
    BusinessTravel +
    MaritalStatus +
    NumCompaniesWorked +
    TotalWorkingYears +
    TrainingTimesLastYear +
    YearsAtCompany +
    YearsWithCurrManager +
    EnvironmentSatisfaction +
    JobSatisfaction
  ,
  family = "binomial",
  data = downsample_prod
)

# Marginal Effects
coefs = 
  tidy(f_log_prod) %>% 
  mutate(odds = exp(estimate),
         odds_mfx = odds - 1)

coefs

mfx = margins(f_log_prod)

summary(mfx)

summary(f_log_prod)

summary(f)

### Decision Trees ----

### Step 5: Feature & Model Selection ----

# Initial Model
f_dt = train(
  Attrition ~ .,
  data = downsample_df,
  method = "rpart",
  tuneGrid = expand.grid(cp = seq(0.001, 0.1, by = 0.01)),  # Tuning the complexity parameter (cp)
  trControl = trainControl(
    method = "cv", 
    number = 10, # 10-fold cross validation
    classProbs = TRUE,
    summaryFunction = twoClassSummary # Use twoClassSummary to compute AUC
  ),
  metric = "ROC" # "ROC" gives us AUC & silences warning about Accuracy
)

f_dt$finalModel

# plot final model
rpart.plot(f_dt$finalModel)

# Variables of Importance
var_imp_dt = varImp(f_dt)
var_imp_dt
plot(var_imp_dt)

# number of unique values equals number of terminal nodes
predict(f_dt, df_train) %>% 
  unique() %>% 
  length()

### Step 6: Model Validation ----

# check cross validation results
f_dt$results # average CV result by cp

f_dt$bestTune # which cp chosen?

f_dt$resample # final model's result for each fold

### Step 7: Predictions and Conclusions ----

p_dt = predict(f_dt, df_test, type = "prob") #predict based on test data

roc_dt_test = roc(df_test$Attrition, p_dt$Yes)

plot(roc_dt_test, main = "ROC Curve for Decision Tree Model")

roc_dt_test$auc

### Extending Decision Trees ----

### Step 5: Feature & Model Selection ----

# bagged decision trees
bagged_model <- 
  train(
    Attrition ~ ., # formula
    data = downsample_df,
    method = "treebag",      # Bagged decision tree method
    trControl = trainControl(
      method = "cv", number = 10, # 10-fold cross validation
      classProbs = TRUE,  # Enable probability predictions
      summaryFunction = twoClassSummary  # Use twoClassSummary to compute AUC
    ),
    metric = "ROC" # "ROC" gives us AUC & silences warning about Accuracy
  ) 

bagged_model # summary

bagged_model$finalModel # number of trees

bagged_model$resample # cross validation results

var_imp_bt <- varImp(bagged_model) # variable importance

var_imp_bt

plot(var_imp_bt)

# fit a model with fewer trees
bagged_model_10 <-
  train(
    Attrition ~ ., # formula
    data = downsample_df,
    method = "treebag",      # Bagged decision tree method
    nbagg = 10, # Adjust the number of trees to bag (default is 25)
    trControl = trainControl(
      method = "cv", number = 10, # 10-fold cross validation
      classProbs = TRUE,  # Enable probability predictions
      summaryFunction = twoClassSummary  # Use twoClassSummary to compute AUC
    ),
    metric = "ROC" # "ROC" gives us AUC & silences warning about Accuracy
  )   

bagged_model_10 # summary

bagged_model_10$finalModel # number of trees

bagged_model_10$resample # cross validation results

var_imp_bt_10 <- varImp(bagged_model_10) # variable importance

var_imp_bt_10

plot(var_imp_bt_10)

# compare the two
vip_for_plotting <- 
  rbind(
    tibble(
      var = rownames(var_imp_bt$importance),
      var_imp_bt$importance,
      num_trees = 25
    ),
    tibble(
      var = rownames(var_imp_bt_10$importance),
      var_imp_bt_10$importance,
      num_trees = 10
    ),
    tibble(
      var = rownames(var_imp_dt$importance),
      var_imp_dt$importance,
      num_trees = 1
    )
  )

vip_for_plotting |>
  ggplot(aes(y = var, x = Overall, fill = factor(num_trees))) +
  geom_bar(stat = "identity", position = "dodge")

bagged_model$results

results_for_plotting <- 
  rbind(
    bagged_model$results[, -1],
    bagged_model_10$results[, -1],
    f_dt$results[1, -1]
  ) |>
  mutate(
    num_trees = c(25, 10, 1)
  ) |>
  select(
    ROC,
    Sens,
    Spec,
    num_trees
  ) |>
  pivot_longer(-num_trees) 

results_for_plotting |>
  ggplot(aes(x = value, y = name, fill = factor(num_trees))) +
  geom_bar(stat = "identity", position = "dodge") + 
  xlim(0, 1) + 
  ggtitle("Comparing Number of Bagged Trees")


### Step 6: Model Validation ----
#pred_bag <- predict(
#  bagged_model, 
#  predict(med_imp, df_validate), # median imputed from training set
#  type = "prob"
#)

#roc_bag <- roc(
#  df_validate$Attrition,
#  pred_bag$Yes
#)

#plot(roc_bag)

#roc_bag$auc




### Step 7: Predictions and Conclusions ----

p_bt <- predict(bagged_model_10, df_test, type = "prob")  # Predicted probabilities

roc_bt_test <- roc(
  response = df_test$Attrition,    # True labels
  predictor = p_bt$Yes      # Predicted probabilities for the positive class
)

# Plot the ROC curve
plot(roc_bt_test, main = "ROC Curve for Bagged Decision Trees Model")

# Compute the AUC
auc(roc_bt_test)


roc_bt_test$auc

### Random Forest ----

### Step 5: Feature & Model Selection ----
rf_model <- 
  train(
    Attrition ~ ., # formula
    data = downsample_df,
    method = "rf",      # Random Forest method
    trControl = trainControl(
      method = "cv", number = 10, # 10-fold cross-validation
      classProbs = TRUE,  # Enable probability predictions
      summaryFunction = twoClassSummary  # Use twoClassSummary to compute AUC
    ),
    metric = "ROC" # "ROC" gives us AUC & silences warning about Accuracy
  )

rf_model # summary

rf_model$finalModel # details of the Random Forest model

rf_model$resample # cross-validation results

var_imp_rf <- varImp(rf_model) # variable importance

var_imp_rf

plot(var_imp_rf)

# plot error based on number of trees
# can use it to assess number of trees we actually need (to speed up training time)
plot(rf_model$finalModel)

# Tune Random Forest: Adjust the number of trees and try different mtry values
rf_model_tuned <- 
  train(
    Attrition ~ ., # formula
    data = downsample_df,
    method = "rf",      # Random Forest method
    tuneGrid = expand.grid(mtry = c(2, 3, 4)), # Tune mtry 
    ntree = 100, # manually change the number of trees
    trControl = trainControl(
      method = "cv", number = 10, # 10-fold cross-validation
      classProbs = TRUE,  # Enable probability predictions
      summaryFunction = twoClassSummary  # Use twoClassSummary to compute AUC
    ),
    metric = "ROC" # "ROC" gives us AUC & silences warning about Accuracy
  )

plot(rf_model_tuned$finalModel)

rf_model_tuned # summary

rf_model_tuned$finalModel # details of the tuned Random Forest model

rf_model_tuned$resample # cross-validation results

var_imp_rf_tuned <- varImp(rf_model_tuned) # variable importance

var_imp_rf_tuned

plot(var_imp_rf_tuned)

# Compare the two
vip_for_plotting <- 
  rbind(
    tibble(
      var = rownames(var_imp_rf$importance),
      var_imp_rf$importance,
      model = "Default RF"
    ),
    tibble(
      var = rownames(var_imp_rf_tuned$importance),
      var_imp_rf_tuned$importance,
      model = "Tuned RF"
    )
  )

vip_for_plotting |>
  ggplot(aes(y = var, x = Overall, fill = model)) +
  geom_bar(stat = "identity", position = "dodge")

results_for_plotting <- 
  rbind(
    rf_model$results |> mutate(model = "Default RF"),
    rf_model_tuned$results |> mutate(model = "Tuned RF")
  ) |>
  select(
    ROC, Sens, Spec, model
  ) |>
  pivot_longer(-model)

results_for_plotting |>
  ggplot(aes(x = value, y = name, fill = model)) +
  geom_bar(stat = "identity", position = "dodge") + 
  xlim(0, 1) + 
  ggtitle("Comparing Default and Tuned Random Forest")

### Step 6: Model Validation ----

#pred_rf <- predict(
#  rf_model, 
#  predict(med_imp, empl_validate), # median imputed from training set
#  type = "prob"
#)

#roc_rf <- roc(
#  empl_validate$Attrition,
#  pred_rf$Yes
#)

#plot(roc_rf)

#roc_rf$auc

### Step 7: Predictions and Model Evaluation ----

# Predict probabilities for the test set using the tuned RF model
p_rf <- predict(
  rf_model_tuned, 
  df_test, 
  type = "prob"
)

# Compute ROC and AUC for the tuned RF model
roc_rf_test <- roc(
  df_test$Attrition,
  p_rf$Yes
)

cat("AUC for Tuned RF Model:", tuned_test_roc$auc, "\n")
plot(roc_rf_test, main = "ROC Curve - Random Forest Model")

### Model Comparison ----
roc_log_test$auc
roc_dt_test$auc
roc_bt_test$auc
roc_rf_test$auc
