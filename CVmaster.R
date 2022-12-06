CVmaster <- function(classifier, features, label, data, k, loss){
  # reproducibility
  set.seed(123)
  # create cv folds
  folds <- createFolds(data$label, k=k)
  modelQualityCV <- data.frame(fold = rep(NA, k), 
                               Train = rep(NA, k), 
                               Test = rep(NA, k))
  colnames(modelQualityCV) <- c("fold", "Train","Test")
  # for loop processing for each classifier
  for(i in 1:k){
    
    #cv-fold
    print(paste("PROCESSING CLUSTER n?", i, "out of", length(folds)))
    id = folds[[i]]
    train = data[-id,]
    test = data[id,]
    train <- train %>% dplyr::select(label, all_of(features))
    test <- test %>% dplyr::select(label,all_of(features))
    label_loc <- which(colnames(test)==label)
    test_pred_set <- test[,-label_loc]
    train_pred_set <- train[,-label_loc]
    train_label <- train[,label_loc]
    modelQualityCV[i,1] <- paste("fold",i)
    form <- as.formula(paste(label,"~",paste(features, collapse = "+"), sep = " "))
    
    #data modification
    trainset <- train
    testset <- test
    trainset$label[trainset$label=="1"] <- 1
    trainset$label[trainset$label=="-1"] <- 0
    testset$label[testset$label=="1"] <- 1
    testset$label[testset$label=="-1"] <- 0
    trainset_fc <- trainset
    trainset_fc$label <- as.factor(trainset_fc$label)
    testset_fc <- testset
    testset_fc$label <- as.factor(testset_fc$label)
    
    threshold <- 0.5
    
    # various classifiers
    if(classifier == "Logistic Regression"){
      
      lr <- glm(form, data=trainset, family="binomial")
      
      lr_pred_prep_train <- predict(lr, train_pred_set, type="response") 
      lr_pred_train <- ifelse(lr_pred_prep_train > threshold, "1","0")
      cm1 <- confusionMatrix(as.factor(lr_pred_train),
                             as.factor(trainset$label))
      lr_pred_prep_test <- predict(lr, test_pred_set, type="response") 
      lr_pred_test <- ifelse(lr_pred_prep_test > threshold, "1","0")
      cm2 <- confusionMatrix(as.factor(lr_pred_test),
                             as.factor(testset$label))
      
      if(loss == "accuracy"){
        accuracy1 <- round(cm1$overall[1],7)
        modelQualityCV[i,2] <- accuracy1
        colnames(modelQualityCV)[2] <- paste0(classifier," ",
                                              "Train Accuracy")
        accuracy2 <- round(cm2$overall[1],7)
        modelQualityCV[i,3] <- accuracy2
        colnames(modelQualityCV)[3] <- paste0(classifier," ",
                                              "Test Accuracy")
      } else if (loss == "F1"){
        f11 <- round(cm1$byClass[7],7)
        modelQualityCV[i,2] <- f11
        colnames(modelQualityCV)[2] <- paste0(classifier," ",
                                              "Train F1-score")
        f12 <- round(cm2$byClass[7],7)
        modelQualityCV[i,3] <- f12
        colnames(modelQualityCV)[3] <- paste0(classifier," ",
                                              "Test F1-score")
      }
      
    } else if(classifier == "LDA"){ #should not change label as factor for acc, mse compute
      
      lda <- lda(form, data=trainset)
      lda_pred_prep_train <- predict(lda, train_pred_set, type="prob") 
      lda_pred_train <- lda_pred_prep_train$class
      cm1 <- confusionMatrix(lda_pred_train, as.factor(trainset$label))
      lda_pred_prep_test <- predict(lda, test_pred_set, type="prob") 
      lda_pred_test <- lda_pred_prep_test$class
      cm2 <- confusionMatrix(lda_pred_test, as.factor(testset$label))
      
      if(loss == "accuracy"){
        accuracy1 <- round(cm1$overall[1],7)
        modelQualityCV[i,2] <- accuracy1
        colnames(modelQualityCV)[2] <- paste0(classifier," ",
                                              "Train Accuracy")
        accuracy2 <- round(cm2$overall[1],7)
        modelQualityCV[i,3] <- accuracy2
        colnames(modelQualityCV)[3] <- paste0(classifier," ",
                                              "Test Accuracy")
      } else if (loss == "F1"){
        f11 <- round(cm1$byClass[7],7)
        modelQualityCV[i,2] <- f11
        colnames(modelQualityCV)[2] <- paste0(classifier," ",
                                              "Train F1-score")
        f12 <- round(cm2$byClass[7],7)
        modelQualityCV[i,3] <- f12
        colnames(modelQualityCV)[3] <- paste0(classifier," ",
                                              "Test F1-score")
      }
      
    } else if(classifier == "QDA"){ #should not change label as factor for acc, mse compute
      
      qda <- qda(form, data=trainset)
      qda_pred_prep_train <- predict(qda, train_pred_set, type="prob") 
      qda_pred_train <- qda_pred_prep_train$class
      cm1 <- confusionMatrix(qda_pred_train, as.factor(trainset$label))
      qda_pred_prep_test <- predict(qda, test_pred_set, type="prob") 
      qda_pred_test <- qda_pred_prep_test$class
      cm2 <- confusionMatrix(qda_pred_test, as.factor(testset$label))
      
      if(loss == "accuracy"){
        accuracy1 <- round(cm1$overall[1],7)
        modelQualityCV[i,2] <- accuracy1
        colnames(modelQualityCV)[2] <- paste0(classifier," ",
                                              "Train Accuracy")
        accuracy2 <- round(cm2$overall[1],7)
        modelQualityCV[i,3] <- accuracy2
        colnames(modelQualityCV)[3] <- paste0(classifier," ",
                                              "Test Accuracy")
      } else if (loss == "F1"){
        f11 <- round(cm1$byClass[7],7)
        modelQualityCV[i,2] <- f11
        colnames(modelQualityCV)[2] <- paste0(classifier," ",
                                              "Train F1-score")
        f12 <- round(cm2$byClass[7],7)
        modelQualityCV[i,3] <- f12
        colnames(modelQualityCV)[3] <- paste0(classifier," ",
                                              "Test F1-score")
      }
      
    } else if(classifier == "Naive Bayes"){ #should change as.factor!!!***
      
      nb <- naive_bayes(form, data=trainset_fc, laplace=1, usekernel=T) 
      
      nb_pred_prep_train <- predict(nb, train_pred_set, type="prob")
      nb_pred_train <- nb_pred_prep_train[,2]
      nb_pred_train <- ifelse(nb_pred_train > threshold, "1","0")
      cm1 <- confusionMatrix(as.factor(nb_pred_train), as.factor(trainset_fc$label))
      nb_pred_prep_test <- predict(nb, test_pred_set, type="prob")
      nb_pred_test <- nb_pred_prep_test[,2]
      nb_pred_test <- ifelse(nb_pred_test > threshold, "1","0")
      cm2 <- confusionMatrix(as.factor(nb_pred_test), as.factor(testset_fc$label))
      
      if(loss == "accuracy"){
        accuracy1 <- round(cm1$overall[1],7)
        modelQualityCV[i,2] <- accuracy1
        colnames(modelQualityCV)[2] <- paste0(classifier," ",
                                              "Train Accuracy")
        accuracy2 <- round(cm2$overall[1],7)
        modelQualityCV[i,3] <- accuracy2
        colnames(modelQualityCV)[3] <- paste0(classifier," ",
                                              "Test Accuracy")
      } else if (loss == "F1"){
        f11 <- round(cm1$byClass[7],7)
        modelQualityCV[i,2] <- f11
        colnames(modelQualityCV)[2] <- paste0(classifier," ",
                                              "Train F1-score")
        f12 <- round(cm2$byClass[7],7)
        modelQualityCV[i,3] <- f12
        colnames(modelQualityCV)[3] <- paste0(classifier," ",
                                              "Test F1-score")
      }   
      
    } else if(classifier == "Random Forest"){
      
      rf <- randomForest(form, trainset_fc)
      
      rf_pred_prep_train <- predict(rf, train_pred_set, type="prob")
      rf_pred_train <- rf_pred_prep_train[,2]
      rf_pred_train <- ifelse(rf_pred_train > threshold, "1","0")
      cm1 <- confusionMatrix(as.factor(rf_pred_train), as.factor(trainset_fc$label))
      rf_pred_prep_test <- predict(rf, test_pred_set, type="prob")
      rf_pred_test <- rf_pred_prep_test[,2]
      rf_pred_test <- ifelse(rf_pred_test > threshold, "1","0")
      cm2 <- confusionMatrix(as.factor(rf_pred_test), as.factor(testset_fc$label))
      
      if(loss == "accuracy"){
        accuracy1 <- round(cm1$overall[1],7)
        modelQualityCV[i,2] <- accuracy1
        colnames(modelQualityCV)[2] <- paste0(classifier," ",
                                              "Train Accuracy")
        accuracy2 <- round(cm2$overall[1],7)
        modelQualityCV[i,3] <- accuracy2
        colnames(modelQualityCV)[3] <- paste0(classifier," ",
                                              "Test Accuracy")
      } else if (loss == "F1"){
        f11 <- round(cm1$byClass[7],7)
        modelQualityCV[i,2] <- f11
        colnames(modelQualityCV)[2] <- paste0(classifier," ",
                                              "Train F1-score")
        f12 <- round(cm2$byClass[7],7)
        modelQualityCV[i,3] <- f12
        colnames(modelQualityCV)[3] <- paste0(classifier," ",
                                              "Test F1-score")
      }   
      
    } else if(classifier == "KNN"){
      set.seed(123)
      knn_part <- partition(trainset_fc$label, p = c(tset = 0.9, vset = 0.1))
      train_knn <- trainset_fc[knn_part$tset, ]
      valid_knn <- trainset_fc[knn_part$vset, ]
      train_pred_set <- train_knn[,-label_loc]
      valid_pred_set <- valid_knn[,-label_loc]
      train_label <- train_knn[,label_loc]
      
      knn_pred_prep_train <- knn(train_pred_set, valid_pred_set, train_label, k=3)
      cm1 <- confusionMatrix(knn_pred_prep_train, as.factor(valid_knn$label))
      knn_pred_prep_test <- knn(train_pred_set, test_pred_set, train_label, k=3)
      cm2 <- confusionMatrix(knn_pred_prep_test, as.factor(testset_fc$label))
      
      if(loss == "accuracy"){
        accuracy1 <- round(cm1$overall[1],7)
        modelQualityCV[i,2] <- accuracy1
        colnames(modelQualityCV)[2] <- paste0(classifier," ",
                                              "Train Accuracy")
        accuracy2 <- round(cm2$overall[1],7)
        modelQualityCV[i,3] <- accuracy2
        colnames(modelQualityCV)[3] <- paste0(classifier," ",
                                              "Test Accuracy")
      } else if (loss == "F1"){
        f11 <- round(cm1$byClass[7],7)
        modelQualityCV[i,2] <- f11
        colnames(modelQualityCV)[2] <- paste0(classifier," ",
                                              "Train F1-score")
        f12 <- round(cm2$byClass[7],7)
        modelQualityCV[i,3] <- f12
        colnames(modelQualityCV)[3] <- paste0(classifier," ",
                                              "Test F1-score")
      }   
    }
  }
  print(modelQualityCV)
}