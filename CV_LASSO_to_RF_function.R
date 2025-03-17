
CV_LASSO_to_RF<-function(data,outcome,
                          MTRY=NULL,NTREE=c(100,500),NODESIZE=c(5,10,15)){
  
  df_name<-deparse(substitute(data))
  
  # First randomizing order of rows before creating folds (in case ID provides meaningful info)  
  randomrowdf <- data[sample(nrow(data), replace=F),]
  
  # break data frame into 4 outer folds by indexing data from 1 to 4
  outerfoldindex <- cut(seq(1,nrow(randomrowdf)), breaks=4, labels=F) # Creates the indices used in for loop below 
  
  # Empty data frame to store predictive performance of entire model on each of the four outer folds
  outerfold_rf_params_metrics<- data.frame(matrix(NA,nrow = 4, ncol = 17))
  colnames(outerfold_rf_params_metrics)<-c("Dataset","Fold","MSE_RF","rsq_RF",
                                           "mtry","ntree","nodesize",
                                           "MSE_LASSO","rsq_LASSO",
                                           "prop_od_captured_top_5th_p",
                                           "prop_od_captured_top_10th_p",
                                           "prop_od_captured_top_15th_p",
                                           "prop_od_captured_top_20th_p",
                                           "BPR_top_5th_p",
                                           "BPR_top_10th_p",
                                           "BPR_top_15th_p",
                                           "BPR_top_20th_p")
  # Object to store variable importance on each of the outer folds
  outerimp <- vector(mode = 'list',length = 4)
  
  # Object to store outer prediction results
  outerpredict <- vector(mode = 'list',length = 4)
  
  # Object to store coefficients that remain after regularization for each outer fold
  thenonzerocoef <- vector(mode = 'list',length = 4)
  
  # Object to store y values from testing set for each outer fold
  yactual <- vector(mode = 'list',length = 4)
  
  # Object to store lambda that minimizes cross-validated MSE 
  MinLambda <- data.frame(matrix(NA, nrow = 4, ncol = 1))
  colnames(MinLambda)<-"Min_lambda"
  # Create object to store parameters from RF w/ lowest MSE from each of the 10 inner folds
  # Will select mode mtry and ntree FROM THIS
  rfparamssaved <- data.frame(matrix(NA, nrow = 10, ncol = 3))
  colnames(rfparamssaved)<-c("mtry","ntree","nodesize")
  
  #create list to save each outer rf model
  outer_rf_model<-vector(mode = 'list',length = 4)
  
  
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
    yactual[[i]] <- ytest <- as.matrix(test[,outcome])
    
    # Training LASSO Regression on training sets created by inner folds and evaluating on inner fold testing sets
    # Unlike RF (which is implemented manually via function) glmnet has built-in CV functionality
    cv.fit <- glmnet::cv.glmnet(x=xtrain,y=ytrain, alpha=1, type.measure="mse", foldid=foldstrain, family="gaussian")
    
    # Saving value of lambda that minimizes cross-validated MSE from inner folds
    MinLambda[i,1] <- cv.fit$lambda.min
    
    # Predictive performance of LASSO Regression in outer fold test set
    predsLASSO <- predict(cv.fit, xtest, type = "response", s="lambda.min")
    
    # Show list of coefficients that do/don't shrink to zero at optimal cross-validated lambda
    cfs <- coef(cv.fit, s="lambda.min")
    print(paste0(df_name,": fold number: ",i))
    print(Sys.time())
    
    # select coefficients that are nonzero after shrinkage
    nonzero <- (rownames(cfs)[abs(cfs[,1])>0])
    #Removing Intercept if it is nonzero
    nonzero<-nonzero[!nonzero%in%"(Intercept)"]
    # Saving list of nonzero coefficients for all outer folds
    thenonzerocoef[[i]] <- nonzero
    
    #Adaptively specifying MTRY candidate values based on p predictors selected by LASSO
    if(is.null(MTRY)){MTRY_op<-floor((length(nonzero))/3)+c(-6,-3,0,3,6)[floor(
        (length(nonzero))/3)+c(-6,-3,0,3,6)>1]}else{
          MTRY_op<-MTRY
        }
    
    # Adding outcome to variable list
    nonzero <- c(nonzero, outcome)
    
    # setting up training and testing sets for outer folds EXCLUDING coeffs shrunk by LASSO
    train0 <- train[,colnames(train) %in% nonzero]
    test0 <- test[,colnames(test) %in% nonzero]
    
    xtrain0 <- as.matrix(train0[,!colnames(train0) %in% outcome])
    xtest0 <- as.matrix(test0[,!colnames(test0) %in% outcome])
    
    
    # Initializing the inner loop - repeat 10 times for EVERY outer fold
    # (Did so for LASSO using glmnet built-in funct above)
    # NOW inner fold hyperparameter tuning for Random Forest algorithm using own function
    
    for (j in 1:10) {
      
      # select the training and test data for this fold
      innerindex <- which(foldstrain==j, arr.ind=T)
      
      innertrain <- train0[-innerindex, ]
      innertest <- train0[innerindex, ]
      
      
      # separating dependent and independent vars - training set
      innerxtrain <- as.matrix(
        innertrain[,!colnames(innertrain) %in% outcome])
      innerytrain <- as.matrix(innertrain[,outcome])
      
      # separating dependent and  independent vars - testing set
      innerxtest <- as.matrix(
        innertest[,!colnames(innertest) %in% outcome])
      #create vector of outcome values 
      innerytest <- as.matrix(innertest[,outcome])
      
      
      # Tuning Random Forest
      # fit random forest
      # ntree is the number of trees grown (generally, larger is better w/ computation tradeoff...)
      # mtry is the number of covariates selected as candidates for each split
      
      rf_param_candidates<-expand.grid(mtry=MTRY_op,ntree=NTREE,
                                       nodesize=NODESIZE)
      # Applying hyperparameter range to RF
      RFModels <- mapply(RFFit, mtry = rf_param_candidates$mtry, 
                         ntree = rf_param_candidates$ntree,
                         nodesize=rf_param_candidates$nodesize,
                         MoreArgs=list(x = innerxtrain, y = innerytrain),
                         SIMPLIFY = F)
      
      
      # Then predict overdose deaths with these hyperparameters
      predictions <- sapply(X=RFModels, FUN=predict, newdata = innerxtest)
      
      # Setting column names for rfparam object
      rfparam <- data.frame(matrix(NA,nrow = nrow(rf_param_candidates), ncol = 4))
      colnames(rfparam) <- c("MSE","mtry","ntree","nodesize")
      
      # Saving MSE/mtry/ntree for each algorithm
      for(h in 1:ncol(predictions)){
        
        # Store MSE results and hyperparameters in data frame together
        rfparam[h,1] <- mean((predictions[,h] - innerytest)^2)
        rfparam[h,2] <- RFModels[[h]]$mtry
        rfparam[h,3] <- RFModels[[h]]$ntree
      }
      rfparam[,4] <- rf_param_candidates$nodesize
      
      #pulling out best ntree, mtry, and nodesize values based on lowest MSE
      rfparam_best<-subset(rfparam, MSE == min(MSE))
      
      # saving ideal hyperparameters (that minimize CV MSE) from each inner fold and saving
      rfparamssaved[j, 1] <- rfparam_best$mtry
      rfparamssaved[j, 2] <- rfparam_best$ntree
      rfparamssaved[j, 3] <- rfparam_best$nodesize
      
      
    }
    
    # Take mode to find optimal configuration of hyperparams for each ind outerfold (rfparammssaved is 10x3 df)
    mtry <- Mode_w_random_tie_break(rfparamssaved[ , 1])
    ntree <- Mode_w_random_tie_break(rfparamssaved[ , 2])
    nodesize <- Mode_w_random_tie_break(rfparamssaved[ , 3])
    
    # Save the optimal hyperparameters for EACH outer fold
    print(mtry)
    print(ntree)
    print(nodesize)
    
    # Rerun Random Forest evaluated on outer fold for performance metrics
    rf_outer <- randomForest(x=xtrain0, y=ytrain, ntree=ntree, mtry=mtry,nodesize=nodesize,
                             keep.forest=T,importance=T)
    
    #save outer rf model into list
    outer_rf_model[[i]]<-rf_outer
    
    # Variable importance results over each outer fold
    outerimp[[i]] <-importance(rf_outer)
    
    # Predictions based on test set
    outerpredict[[i]] <- outerpred <- predict(rf_outer, newdata = xtest0, type = "response")
    
    #Table saving all values for each fold
    outerfold_rf_params_metrics[i,1]<-df_name
    outerfold_rf_params_metrics[i,2]<-i
    outerfold_rf_params_metrics[i,3]<-mean((outerpred-ytest)^2)
    outerfold_rf_params_metrics[i,4]<-sum((outerpred-mean(ytest))^2)/sum((ytest-mean(ytest))^2)
    outerfold_rf_params_metrics[i,5]<-mtry
    outerfold_rf_params_metrics[i,6]<-ntree
    outerfold_rf_params_metrics[i,7]<-nodesize
    outerfold_rf_params_metrics[i,8]<-mean((predsLASSO-ytest)^2)
    outerfold_rf_params_metrics[i,9]<-sum((predsLASSO-mean(ytest))^2)/sum((ytest-mean(ytest))^2)
    outerfold_rf_params_metrics[i,10]<-Percent_OD_captured(preds = outerpred,
                                                           yactual = ytest,
                                                           action_percent = 5)[1]
    outerfold_rf_params_metrics[i,11]<-Percent_OD_captured(preds = outerpred,
                                                           yactual = ytest,
                                                           action_percent = 10)[1]
    outerfold_rf_params_metrics[i,12]<-Percent_OD_captured(preds = outerpred,
                                                           yactual = ytest,
                                                           action_percent = 15)[1]
    outerfold_rf_params_metrics[i,13]<-Percent_OD_captured(preds = outerpred,
                                                           yactual = ytest,
                                                           action_percent = 20)[1]
    outerfold_rf_params_metrics[i,14]<-Best_possible_reach(preds = outerpred,
                                                           yactual = ytest,
                                                           action_percent = 5)[1]
    outerfold_rf_params_metrics[i,15]<-Best_possible_reach(preds = outerpred,
                                                           yactual = ytest,
                                                           action_percent = 10)[1]
    outerfold_rf_params_metrics[i,16]<-Best_possible_reach(preds = outerpred,
                                                           yactual = ytest,
                                                           action_percent = 15)[1]
    outerfold_rf_params_metrics[i,17]<-Best_possible_reach(preds = outerpred,
                                                           yactual = ytest,
                                                           action_percent = 20)[1]
    
    
  }
  #training model on all training data
  rfparam_best_overall<-subset(outerfold_rf_params_metrics, MSE_RF == min(MSE_RF,na.rm = T))
  mtry_overall<-rfparam_best_overall$mtry
  ntree_overall<-rfparam_best_overall$ntree
  nodesize_overall<-rfparam_best_overall$nodesize
  nonzero_overall<-thenonzerocoef[[rfparam_best_overall$Fold]]
  full_x_train<-as.matrix(randomrowdf[,colnames(randomrowdf) %in% nonzero_overall])
  full_y_train<-as.matrix(randomrowdf[[outcome]])
  full_training_rf_model<-randomForest(mtry = mtry_overall,ntree = ntree_overall,
                                    nodesize = nodesize_overall,
                                    x=full_x_train,y = full_y_train,keep.forest=T,
                                    importance=T)
  
  #Saving all important objects
  lasso_to_rf_output<-list(full_model=full_training_rf_model,
                           full_model_importance=importance(full_training_rf_model),
                           full_model_vars=nonzero_overall,
                           full_model_hyper_params=list(mtry=mtry_overall,
                                                        ntree=ntree_overall,
                                                        nodesize=nodesize_overall),
                           fold_details=list(best_fold=rfparam_best_overall$Fold,
                                             fold_predictions=outerpredict,
                                             fold_y_actuals=yactual,
                                             fold_vars=thenonzerocoef,
                                             fold_metrics=outerfold_rf_params_metrics,
                                             fold_min_lambda=MinLambda,
                                             fold_models=outer_rf_model,
                                             fold_importances=outerimp
                                             )
  )
  
  return(lasso_to_rf_output)
}

# RFFit function for CV_LASSO_to_RF function
RFFit <- function(mtry, ntree, nodesize, x, y){
  randomForest(x=x, y=y, mtry = mtry, ntree = ntree,nodesize = nodesize)
}

# mode with tie break function for CV_LASSO_to_RF function
Mode_w_random_tie_break<-function(x){
  value_counts<-table(x)
  max_freq<-max(value_counts)
  mode_values<-names(value_counts[value_counts==max_freq])
  if (length(mode_values>1)){
    random_index<-sample(1:length(mode_values),size=1)
    return(as.numeric(mode_values[random_index]))
  } else {return(as.numeric(mode_values[1]))
    
  }
}