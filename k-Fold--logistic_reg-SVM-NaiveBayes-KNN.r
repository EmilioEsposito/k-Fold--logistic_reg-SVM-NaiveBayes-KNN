# You need to install package 'FNN'
library(FNN)
library(e1071)
library(plyr)
options(scipen = 999)

# utility function for import from csv file
import.csv <- function(filename) {
    return(read.csv(filename, sep = ",", header = TRUE))
}

# utility function for export to csv file
write.csv <- function(ob, filename) {
    write.table(ob, filename, quote = FALSE, sep = ",", row.names = FALSE)
}

#PART 1

# Logistic regression model
# Assumes the last column of data is the output dimension, and that it's numeric binary
get_pred_logreg <- function(train,test){
  
  #get number of columns
  nf <- ncol(train)
  
  #get independent variable for training
  ytrain <- train[ , nf]
  #colnames(ytrain) <- colnames(train)[ncol(ytrain)]
  
  #get dependent variables for training
  xtrain <- data.frame(train[ , -nf])
  colnames(xtrain) <- colnames(train)[1:ncol(xtrain)]
  
  logreg_mod <- glm(ytrain ~ ., data = xtrain, family = "binomial")
  
  #get dependent variables for testing
  xtest <- data.frame(test[ , -nf])
  colnames(xtest) <- colnames(test)[1:ncol(xtest)]
  
  #get predicted values based on test df
  pred <- predict(logreg_mod, xtest, type = "response")
  
  #get true values of test
  truth <- test[,nf]
  
  pred_truth <- cbind(pred, truth)
  
  return(pred_truth)
}

# SVM model
# Assumes the last column of data is the output dimension
# Assumes numeric binary output column (0 and 1)
get_pred_svm <- function(train,test){
  
  #get number of columns
  nf <- ncol(train)
  
  #get independent variable for training
  ##convet to factor
  ytrain <- as.factor(train[ , nf])
  #colnames(ytrain) <- colnames(train)[ncol(ytrain)]
  
  #get dependent variables for training
  xtrain <- data.frame(train[ , -nf])
  colnames(xtrain) <- colnames(train)[1:ncol(xtrain)]
  
  svm_mod <- svm(ytrain ~., data = xtrain, probability = TRUE)
  
  #get dependent variables for testing
  xtest <- data.frame(test[ , -nf])
  colnames(xtest) <- colnames(test)[1:ncol(xtest)]
  
  #get predicted values based on test df
  pred <- predict(svm_mod, xtest, probability = TRUE)
  #extract probabilities
  pred <- attributes(pred)
  pred <- pred$probabilities
  #only keep probability of "1" factor
  pred <- pred[,"1"]

  #get true values of test
  truth <- test[,nf]
  
  pred_truth <- cbind(pred, truth)
  
  return(pred_truth)
}

# Naive Bayes model
# Assumes the last column of data is the output dimension
# Assumes numeric binary output column (0 and 1)
get_pred_nb <- function(train,test){
  
  #get number of columns
  nf <- ncol(train)
  
  #get independent variable for training
  ##convet to factor
  ytrain <- as.factor(train[ , nf])
  #colnames(ytrain) <- colnames(train)[ncol(ytrain)]
  
  #get dependent variables for training
  xtrain <- data.frame(train[ , -nf])
  colnames(xtrain) <- colnames(train)[1:ncol(xtrain)]
  
  nb_mod <- naiveBayes(ytrain ~., data = xtrain)
  
  #get dependent variables for testing
  xtest <- data.frame(test[ , -nf])
  colnames(xtest) <- colnames(test)[1:ncol(xtest)]
  
  #get predicted values based on test df
  pred <- predict(nb_mod, xtest, type = "raw")
  #only keep probability of factor "1"
  pred <- pred[,"1"]
  
  #get true values of test
  truth <- test[,nf]
  
  pred_truth <- cbind(pred, truth)
  
  return(pred_truth)
}

# knn
# Assumes the last column of data is the output dimension
# Assumes numeric binary output column (0 and 1)
get_pred_knn <- function(train, test, k){
  nf <- ncol(train)
  input <- train[,-nf]
  query <- test[,-nf]
  my.knn <- get.knnx(input,query,k=k) # Get k nearest neighbors
  nn.index <- my.knn$nn.index
  pred <- rep(NA,nrow(test))
  truth <- rep(NA,nrow(test))
  for (ii in 1:nrow(test)){
    neighborIndices <- nn.index[ii,]
    neighborYs <- train[neighborIndices, nf]
    pred[ii] <- mean(neighborYs)
    
    #get true values of test
    truth[ii] = test[ii, nf]
  }
  pred_truth <- cbind(pred, truth)
  
  
  return(pred_truth)  
}


# Default predictor model
# Assumes the last column of data is the output dimension
get_pred_default <- function(train,test){
  # Your implementation goes here
  #find average of output ind var (last col)
  pred <- mean(train[ ,ncol(train)])
  truth <- test[,ncol(test)]
  pred_truth <- cbind(pred, truth)
  return(pred_truth)
}

#PART 2

#kfold cross validation
do_cv_class <- function(df, num_folds, model_name) {
  
  #randomize the rows
  set.seed(2) #guarantees same random order for each model
  rand.index.rows <- sample(nrow(df),size=nrow(df),replace=FALSE)
  df <- df[rand.index.rows, ]
  
  #get number of columns
  nf <- ncol(df)
  
  #if output is factor, convert to numeric binary (0,1)
  if(is.factor(df[ , nf])) {
    df[ , nf] <- as.numeric(df[ , nf])
    
    ##first level will be mapped to 1, second level will be 0
    df[ , nf] <- mapvalues(df[ , nf], c(1,2), c(1,0))
  }
  
  #make empty vector to store all pred_truth
  pred_truth_all <- data.frame()

  #subset the df into test and train
  for(i in 1:num_folds) {
    #set starting row to begin subset
    if(i==1) {
      start.row.test <- 1
      end.row.test <- 0
    } else {
      start.row.test <- end.row.test + 1
    }

    #set endrow
    #last num_folds fold will be slightly larger if not divisible by num_folds
    end.row.test <- min(start.row.test + floor(nrow(df)/num_folds) - 1,nrow(df))
    
    #if it's last fold, include up to last row in test set
    if(i==num_folds) {
      end.row.test = nrow(df)
    }
    #subset the df
    test <- df[c(start.row.test:end.row.test), ]
    train <- df[-c(start.row.test:end.row.test), ]
    
    #determine model from model_name string
    if(grepl(pattern = "nn", x= model_name)) {
      k <- as.numeric(strsplit(model_name,"nn"))
      
      #run the model 
      pred_truth <- get_pred_knn(train,test,k)
    }
    else {
      #run the model 
      model <- get(paste("get_pred_", model_name, sep = ""), mode="function") 
      pred_truth <- model(train, test)
    }
    
    #accumulate results for all folds
    pred_truth_all <- rbind(pred_truth_all, pred_truth)
  }
  
  return(pred_truth_all)
}


#PART 3
# set default cutoff to 0.5 (this can be overridden at invocation time)
get_metrics <- function(pred_truth, cutoff=0.5) {
  
  #transform probabilities into 0/1
  pred_truth[,"pred"] <- aaply(.data = pred_truth[,"pred"], .margins = 1,.fun = function(x) if(x>=cutoff) 1 else 0 )
  
  #find true positives
  pred_truth$tp <- with(pred_truth, as.numeric(pred==1 & truth==1))
  
  #find true negatives
  pred_truth$tn <- with(pred_truth, as.numeric(pred==0 & truth==0))
  
  #find false positives
  pred_truth$fp <- with(pred_truth, as.numeric(pred==1 & truth==0))
  
  tpr <- with(pred_truth, sum(tp)/sum(truth))
  fpr <- with(pred_truth, sum(fp)/sum(!truth))
  acc <- with(pred_truth, sum(tn+tp)/nrow(pred_truth))
  precision <- with(pred_truth, sum(tp)/sum(tp+fp)) 
  recall <- tpr

  metrics <- data.frame(tpr=tpr, fpr=fpr, acc=acc, precision=precision, recall=recall)
  return(metrics)
}


#PART 4 A

wine <- import.csv("wine.csv")

#Run different knn to find optimal k
get_metrics(do_cv_class(wine, 10, "1nn"))
get_metrics(do_cv_class(wine, 10, "3nn"))
get_metrics(do_cv_class(wine, 10, "5nn"))
get_metrics(do_cv_class(wine, 10, "7nn"))
get_metrics(do_cv_class(wine, 10, "10nn"))
get_metrics(do_cv_class(wine, 10, "20nn"))
get_metrics(do_cv_class(wine, 10, "100nn"))

#run knn for k 1:25 and calc RMSE and Accuracy
all_mean_rmse <- data.frame()
all_acc <- data.frame()
for(k in 1:25) {
  knn <- paste(k,"nn", sep="")
  result <- do_cv_class(wine, 10, knn)
  
  #calc RMSE
  error <- result$pred-result$truth
  mean_rmse <- mean((sum(error*error)/nrow(result))^.5)
  mean_rmse <- cbind(k, mean_rmse)
  all_mean_rmse <- rbind(all_mean_rmse, mean_rmse)
  
  #calc accuracy
  metrics <- get_metrics(result)
  acc <- cbind(k, accuracy=metrics$acc)
  all_acc <- rbind(all_acc, acc)
}

#plot accuracy  against k
library(ggplot2)
ggplot(data = all_acc, aes(x=k, y=accuracy)) + geom_point() + geom_line() +
  ggtitle("Accuracy vs k for K-Nearnest-Neighbors Model")+
  theme(text = element_text(size=20))

#plot RMSE  against k
library(ggplot2)
ggplot(data = all_mean_rmse, aes(x=k, y=mean_rmse)) + geom_point() + geom_line() +
  ggtitle("RMSE vs k for K-Nearnest-Neighbors Model")+
  theme(text = element_text(size=20))

#find k where accuracy is maximized
all_acc[which.max(all_acc$accuracy),]

#find k where error is minimized
all_mean_rmse[which.min(all_mean_rmse$mean_rmse),]



#PART 4 B
#3 parametric models are: Logistic Regression, Naive Bayes, and SVM

#run 10fold do_cv_class and cal metrics using 3 parametric models and default model
#using default cutoff of 0.5
logreg_metrics <- get_metrics(do_cv_class(wine, 10, "logreg"))
svm_metrics <- get_metrics(do_cv_class(wine, 10, "svm"))
nb_metrics <- get_metrics(do_cv_class(wine, 10, "nb"))
default_metrics <- get_metrics(do_cv_class(wine, 10, "default"))

#make a summary data frame of metrics
summary_b <- rbind(logreg_metrics, svm_metrics, nb_metrics, default_metrics)
row.names(summary_b) <- c("logreg","svm", "nb", "default")
round(summary_b,3)
#svm has highest accuracy

#PART 4 C

SevenNN_metrics <- get_metrics(do_cv_class(wine, 10, "7nn"))
summary_c <- rbind(logreg_metrics, svm_metrics, nb_metrics, default_metrics, SevenNN_metrics)
row.names(summary_c) <- c("logreg","svm", "nb", "default", "7nn")
round(summary_c,3)

