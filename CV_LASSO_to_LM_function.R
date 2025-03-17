
CV_LASSO_to_LM<-function(data,outcome){
  
  df_name<-deparse(substitute(data))
  
  # First randomizing order of rows before creating folds (in case ID provides meaningful info)  
  randomrowdf <- data[sample(nrow(data), replace=F),]
  
  # break data frame into 4 outer folds by indexing data from 1 to 4
  outerfoldindex <- cut(seq(1,nrow(randomrowdf)), breaks=4, labels=F) # Creates the indices used in for loop below 
  
  # Empty data frames to store predictive performance of LASSO model only on each of the four outer folds
  rsqLASSO <- data.frame(matrix(NA,nrow = 4, ncol = 1))
  MSELASSO <- data.frame(matrix(NA,nrow = 4, ncol = 1))
  
  # Empty data frames to store predictive performance of entire model on each of the four outer folds
  MSE <- data.frame(matrix(NA,nrow = 4, ncol = 1))
  rsq <- data.frame(matrix(NA,nrow = 4, ncol = 1))
  fold_lm_metrics<- data.frame(matrix(NA,nrow = 4, ncol = 15))
  colnames(fold_lm_metrics)<-c("Dataset","Fold","MSE_LM","rsq_LM",
                               "minLambda_LASSO","MSE_LASSO","rsq_LASSO",
                               "prop_od_captured_top_5th_p",
                               "prop_od_captured_top_10th_p",
                               "prop_od_captured_top_15th_p",
                               "prop_od_captured_top_20th_p",
                               "BPR_top_5th_p",
                               "BPR_top_10th_p",
                               "BPR_top_15th_p",
                               "BPR_top_20th_p")
  
  # Object to store prediction results for each fold
  outerpredict <- list()
  
  # Object to store coefficients that remain after regularization for each fold
  thenonzerocoef <- list()
  
  # Object to store y values from testing set for each outer fold
  yactual <- list()
  
  # Object to store lambda that minimizes cross-validated MSE 
  MinLambda <- data.frame(matrix(NA, nrow = 4, ncol = 1))
  colnames(MinLambda)<-"Min_lambda"
  
  #create list to save each outer rf model
  lm_fold_models<-list()
  
  for (i in 1:4) {
    
    # select the training and test data for this fold based on outer fold index
    testindex <- which(outerfoldindex==i, arr.ind=T)
    
    train <- randomrowdf[-testindex,]
    test <- randomrowdf[testindex,]
    
    # Create the 10 inner folds (within each outer fold)
    foldstrain <- cut(seq(1,nrow(train)), breaks=10, labels=F)
    
    
    # separating dependent and independent variables (for glmnet) - training set
    xtrain <- as.matrix(train[,!colnames(train) %in% outcome])
    ytrain <- as.matrix(train[,outcome])
    
    # separating dependent and independent variables - testing set
    xtest <- as.matrix(test[,!colnames(test) %in% outcome])
    ytest <- as.matrix(test[,outcome])
    
    # Training LASSO Regression on training sets created by inner folds and evaluating on inner fold testing sets
    
    cv.fit <- glmnet::cv.glmnet(x=xtrain,y=ytrain, alpha=1, type.measure="mse", 
                                foldid=foldstrain, family="gaussian")
    
    MinLambda[i,1] <- cv.fit$lambda.min
    # Saving value of lambda that minimizes cross-validated MSE from inner folds
    
    # Predictive performance of LASSO Regression in outer fold test set
    predsLASSO <- predict(cv.fit, xtest, type = "response",s="lambda.min")
    
    # Show list of coefficients that do/don't shrink to zero at optimal cross-validated lambda
    cfs <- coef(cv.fit, s="lambda.min")
    #print(cfs)
    print(paste0(df_name,": fold number: ",i))
    print(Sys.time())
    
    # select coefficients that are nonzero after shrinkage
    nonzero <- (rownames(cfs)[abs(cfs[,1])>0])
    #Removing Intercept if it is nonzero
    nonzero<-nonzero[!nonzero%in%"(Intercept)"]
    # Saving list of nonzero coefficients for all outer folds
    thenonzerocoef[[i]] <- nonzero
    
    # Adding outcome to variable list
    nonzero <- c(nonzero, outcome)
    
    # setting up training and testing sets for outer folds EXCLUDING coeffs shrunk by LASSO
    train0 <- train[,colnames(train) %in% nonzero]
    test0 <- test[,colnames(test) %in% nonzero]
    
    #fit lm model on coeffs selected by LASSO
    lm_model<-glm(as.formula(paste(outcome,"~.")),family = gaussian(),
                  data = train0)
    #save outer lm model into list
    lm_fold_models[[i]]<-lm_model
    
    # Predictions based on test set
    outerpred <- predict(lm_model, newdata = test0, type = "response")
    # Saving results for calibration plot
    outerpredict[[i]] <- outerpred
    yactual[[i]] <- ytest
    
    #Table saving all values for each fold
    fold_lm_metrics[i,1]<-df_name
    fold_lm_metrics[i,2]<-i
    fold_lm_metrics[i,3]<-mean((outerpred-ytest)^2)
    fold_lm_metrics[i,4]<-sum((outerpred-mean(ytest))^2)/
      sum((ytest-mean(ytest))^2)
    fold_lm_metrics[i,5]<-cv.fit$lambda.min
    fold_lm_metrics[i,6]<-mean((predsLASSO-ytest)^2)
    fold_lm_metrics[i,7]<-sum((predsLASSO-mean(ytest))^2)/
      sum((ytest-mean(ytest))^2)
    fold_lm_metrics[i,8]<-Percent_OD_captured(
      preds = outerpred,yactual = ytest,action_percent = 5)[1]
    fold_lm_metrics[i,9]<-Percent_OD_captured(
      preds = outerpred,yactual = ytest,action_percent = 10)[1]
    fold_lm_metrics[i,10]<-Percent_OD_captured(
      preds = outerpred,yactual = ytest,action_percent = 15)[1]
    fold_lm_metrics[i,11]<-Percent_OD_captured(
      preds = outerpred,yactual = ytest,action_percent = 20)[1]
    fold_lm_metrics[i,12]<-Best_possible_reach(
      preds = outerpred,yactual = ytest,action_percent = 05)[1]
    fold_lm_metrics[i,13]<-Best_possible_reach(
      preds = outerpred,yactual = ytest,action_percent = 10)[1]
    fold_lm_metrics[i,14]<-Best_possible_reach(
      preds = outerpred,yactual = ytest,action_percent = 15)[1]
    fold_lm_metrics[i,15]<-Best_possible_reach(
      preds = outerpred,yactual = ytest,action_percent = 20)[1]
  }
  #training model on all training data
  best_overall<-subset(fold_lm_metrics, MSE_LM == min(MSE_LM))
  nonzero_overall<-thenonzerocoef[[best_overall$Fold]]
  
  full_train<-randomrowdf[,colnames(randomrowdf) %in% c(nonzero_overall,outcome)]
  
  full_training_lm_model<-glm(as.formula(paste(outcome,"~.")),family = gaussian(),
                              data = full_train)
  
  #Saving all important objects
  lasso_to_lm_output<-list(full_model=full_training_lm_model,
                full_model_vars=nonzero_overall,
                fold_details=list(best_fold=best_overall$Fold,
                                  fold_predictions=outerpredict,
                                  fold_y_actuals=yactual,
                                  fold_vars=thenonzerocoef,
                                  fold_metrics=fold_lm_metrics,
                                  fold_min_lambda=MinLambda,
                                  fold_models=lm_fold_models)
                )
  return(lasso_to_lm_output)
}