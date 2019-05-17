#Predict Loan Status

library(data.table)
train <- fread(file.choose(), stringsAsFactors = T)
#test <- fread(file.choose(), stringsAsFactors = T)

#Deleting Empty rows
train <- head(train,100000)

#Deleting duplicated rows
train <- subset(train, !duplicated(`Loan ID`))

#Formatting Years in current job column
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "8 years"] <- "8"
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "10+ years"] <- "11"
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "3 years"] <- "3"
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "5 years"] <- "5"
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "6 years"] <- "6"
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "7 years"] <- "7"
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "9 years"] <- "9"
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "< 1 year"] <- "0.5"
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "1 year"] <- "1"
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "2 years"] <- "2"
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "4 years"] <- "4"
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "n/a"] <- "11"        #Mode of `Years in current job`

#Encoding categorical variables
train$`Loan Status` <- factor(train$`Loan Status`, labels = c(1,0), levels = c('Fully Paid','Charged Off'))
train$Term <- factor(train$Term, labels = c(1,2), levels = c('Short Term','Long Term'))
train$`Home Ownership` <- factor(train$`Home Ownership`, labels = c(1,2,3,4), levels = c('Home Mortgage','Own Home','Rent','HaveMortgage') )
train$Purpose <- factor(train$Purpose, labels = c(1:17), c('Home Improvements','Debt Consolidation','Buy House','Business Loan','Buy a Car','Educational Expenses','Medical Bills','Take a Trip','major_purchase','moving','vacation','renewable_energy','small_business','wedding','','Other','other'))
#Since 15,16,17 are in to same category
levels(train$Purpose)[levels(train$Purpose) ==  "15"] <- "17"
levels(train$Purpose)[levels(train$Purpose) ==  "16"] <- "17"

#Converting factore into integers
train$`Loan Status` <- as.integer(train$`Loan Status`)
train$Term <- as.integer(train$Term)
train$`Home Ownership` <- as.integer(train$`Home Ownership`)
train$Purpose <- as.integer(train$Purpose)
train$`Years in current job` <- as.integer(train$`Years in current job`)

#Changing Incorrect values in credit score
train[which(train$`Credit Score`>1000),][,6] <- (train[which(train$`Credit Score`>1000),][,6])/10

#################################### Don't run ##########################################################
#Outlier Detection
outlierKD <- function(dt, var) {
  var_name <- eval(substitute(var),eval(dt))
  tot <- sum(!is.na(var_name))
  na1 <- sum(is.na(var_name))
  m1 <- mean(var_name, na.rm = T)
  par(mfrow=c(2, 2), oma=c(0,0,3,0))
  boxplot(var_name, main="With outliers")
  hist(var_name, main="With outliers", xlab=NA, ylab=NA)
  outlier <- boxplot.stats(var_name)$out
  mo <- mean(outlier)
  var_name <- ifelse(var_name %in% outlier, NA, var_name)
  boxplot(var_name, main="Without outliers")
  hist(var_name, main="Without outliers", xlab=NA, ylab=NA)
  title("Outlier Check", outer=TRUE)
  na2 <- sum(is.na(var_name))
  message("Outliers identified: ", na2 - na1, " from ", tot, " observations")
  message("Proportion (%) of outliers: ", (na2 - na1) / tot*100)
  message("Mean of the outliers: ", mo)
  m2 <- mean(var_name, na.rm = T)
  message("Mean without removing outliers: ", m1)
  message("Mean if we remove outliers: ", m2)
  response <- readline(prompt="Do you want to remove outliers and to replace with NA? [yes/no]: ")
  if(response == "y" | response == "yes"){
    dt[as.character(substitute(var))] <- invisible(var_name)
    assign(as.character(as.list(match.call())$dt), dt, envir = .GlobalEnv)
    message("Outliers successfully removed", "\n")
    return(invisible(dt))
  } else{
    message("Nothing changed", "\n")
    return(invisible(var_name))
  }
}

train_data1 <- as.data.frame(train_data)
outlierKD(train_data1, `Annual Income`)               #3047 outliers
outlierKD(train_data1, `Current Loan Amount`)         #9688 outliers
outlierKD(train_data1, `Monthly Debt`)                #2777 outliers
outlierKD(train_data1, `Current Credit Balance`)      #4276 Outliers    Decreased range From 0 - 32878968 to 0 - 747498
outlierKD(train_data1, `Maximum Open Credit`)         #4771 Outliers    Decreased range From 0 - 1539737892 to 0 - 1574980

train$`Annual Income` <- train_data1$`Annual Income`
train$`Current Loan Amount` <- train_data1$`Current Loan Amount`
train$`Monthly Debt` <- train_data1$`Monthly Debt`
train$`Current Credit Balance` <- train_data1$`Current Credit Balance`
train$`Maximum Open Credit` <- train_data1$`Maximum Open Credit`

################################ Accuracy decreasing when we delete outliers ##################################

#Finding Missing Values
sort(sapply(train, function(x) { sum(is.na(x)) }), decreasing=TRUE)

#Note: Credit Score and Annual Income has same number of missing values

############################# Don't run this its taking tooooo long #########################################
#Imputing Missing Values
library(mice)
imputed_data <- mice(train[,c(6,7,17,18,19)], m=5, maxit = 50, method = 'rf', seed = 500)
train[,c(6,7,17,18,19)] <- complete(imputed_data, 2)
############################## Taking toooo long ############################################################

#imputing Na values using Amelia
#install.packages("Amelia")
library(Amelia)
amelia_fit <- amelia(train[,c(4,6,7,11,16,17)], m=5, parallel = "multicore")
train[,c(4,6,7,11,16,17)] <- amelia_fit$imputations[[2]]
########################## Unable to insert values in bancruptcies :-( #######################################

#Finding mode for the bancruptcies
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
mode(train$Bankruptcies)

#Inserting missing values with mode in bancrupcies (here mode is 0)
train$Bankruptcies[is.na(train$Bankruptcies)] <- 0
train$`Tax Liens`[is.na(train$`Tax Liens`)] <- 0

# Removing `Loan ID`` and `Customer ID``
# Removing `Months since last delinquent`` since its has 50% of missing values
# placing target label at last column
train <- train[,c(4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,3)]

#Feature Scaling
train_data <- train
library(caret)
preObj <- preProcess(train_data[, -16], method=c("center", "scale"))
train_data <- predict(preObj, train_data[, -16])
train_data$LoanStatus <- train[,16]

############################################## End of data preparation ###################################

#Splitting train data in to two datasets for validation
library(caTools)
set.seed(123)
split <- sample.split(train_data$LoanStatus, SplitRatio = 0.80)
trainset <- subset(train_data, split == TRUE)
testset <- subset(train_data, split == FALSE)

#Applying kernel PCA for dimensinality reduction
library(kernlab)
kpca = kpca(~., data = trainset, kernel = 'rbfdot', features = 10)
training_set_pca = as.data.frame(predict(kpca, trainset))
training_set_pca$Loan_Status = trainset$Loan_Status
test_set_pca = as.data.frame(predict(kpca, testset))
test_set_pca$Loan_Status = testset$Loan_Status

############################################# Model Preparation starts here #############################################

#Fitting training data to KNN
library(class)
knn_classifier <- knn(train = trainset[,-16], test = testset[,-16], cl = trainset$LoanStatus, k=5, prob = TRUE)

#Fitting training data to Random Forest
library(randomForest)
set.seed(123)
rf_classifier <- randomForest(x = trainset[,-16], y = trainset$LoanStatus, ntree = 500, type = 'classification')

#Fitting training data to SVM
library(e1071)
svm_classifier = svm(formula = LoanStatus ~ .,
                     data = trainset,
                     type = 'C-classification',
                     kernel = 'sigmoid')

#Fitting training data to naive bayes
library(e1071)
nb_classifier = naiveBayes(x = trainset[,-16],
                           y = trainset$LoanStatus)

#Fitting training data to decision tree
library(rpart)
dt_classifier <- rpart(formula = LoanStatus ~ ., data = train_data)

#Fitting training data to XGBoost Model
library(xgboost)
xg_classifier <- xgboost(data = as.matrix(trainset[,-16]), label = trainset$LoanStatus, nrounds = 10)

#Fitting training data to Gradient boost model
library(gbm)
gb_classifier <- gbm(LoanStatus ~ ., data = trainset,distribution = "gaussian",n.trees = 10000, interaction.depth = 4, shrinkage = 0.01)

################################## Artificial Neural network ######################################
library(h2o)
h2o.init(nthreads = -1)
ann_classifier <- h2o.deeplearning(y = 'LoanStatus',
                                   training_frame = as.h2o(trainset),
                                   activation = 'Rectifier',
                                   epochs = 100,
                                   hidden = c(8,8),
                                   train_samples_per_iteration = -2)

################################### Something Went Wrong here########################################################


###########################################################################################
#cross validation
# library(caret)
# train_control<- trainControl(method="cv", number=10, savePredictions = TRUE)
# model<- train(LoanStatus~., data=trainset, trControl=train_control, method="rpart")
############################### Error: undefined columns selected #########################

#Predicting the test results
svm_pred <- predict(svm_classifier, newdata = testset[,-16])
nb_pred <-  predict(nb_classifier, newdata = testset[,-16])
dt_pred <-  predict(dt_classifier, newdata = testset[,-16])
rf_pred <-  predict(rf_classifier, newdata = testset[,-16])
rf_pred <- as.integer(rf_pred)
xg_pred <-  predict(xg_classifier, newdata = as.matrix(testset[,-16]))
xg_pred <- (xg_pred >= 0.5)
n.trees = seq(from=100 ,to=10000, by=100)
gb_pred <- predict(gb_classifier, newdata = testset[,-16], n.trees = n.trees)
#kf_pred <- predict(model, newdata = testset[,-16])

pred <- as.data.frame(predict(ann_classifier, newdata = as.h2o(testset[,-16]))) #Predicting all the target values as 1
#h2o_pred <- ifelse(pred[,3] >0.5,0,1)   #If LoanStatus is 0 and 1
#h2o_pred <- as.integer(h2o_pred)

#confusion matrix                                                           
cm_knn <- table(testset$LoanStatus,knn_classifier)    #72% Accuracy, after deleting outliers 68%
cm_svm <- table(testset$LoanStatus, svm_pred)         #67% Accuracy
cm_nb <- table(testset$LoanStatus, nb_pred)           #6490/9910
cm_dt <- table(testset$LoanStatus, dt_pred)           
cm_rf <- table(testset$LoanStatus, rf_pred)           #72% after deleting outliers
cm_xg <- table(testset$LoanStatus, xg_pred)            
#cm_gb <- table(testset$LoanStatus, gb_pred)         #Gettign Error:all arguments must have the same length
#cm_kf <- table(testset$LoanStatus, kf_pred)            

cm_ann <- table(testset$LoanStatus, h2o_pred)         #77% accuracy after deleting outliers 72%

ann_pred <- round(pred)
ann_pred <- as.integer(ann_pred)
cm_ann <- table(testset$LoanStatus, ann_pred[[1]])

################### or ################################

library(caret)
confusionMatrix(testset$LoanStatus, ann_pred[[1]])   #72%

h2o.shutdown()
