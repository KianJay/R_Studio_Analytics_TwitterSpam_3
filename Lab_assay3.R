#1. Load dataset into R Studio
TwitterSpam <- read.csv("workspace/R/TwitterSpam/TwitterSpam.txt")

# Randomly split the dataset to training dataset and testing dataset with the ratio of 8:2
set.seed(100)
TwitterSpamSplit <- sample(1:nrow(TwitterSpam), 0.8 * nrow(TwitterSpam))
TwitterSpamTrain <- TwitterSpam[TwitterSpamSplit,]
TwitterSpamTest <- TwitterSpam[-TwitterSpamSplit,]

#2.Use training dataset to train a machine learning model with
#   the random forest algorithm for Twitter spam classification 
library(caret) # Import caret library to train the dataset.
RF_model <- train(as.factor(label) ~ ., 
                data = TwitterSpamTrain,
                method='ranger'
                )
RF_model

#3.Use testing dataset to test and evaluate the model trained in step 2 
# and print the confusion matrix.
# Predict the outcome on TwitterSapmTest
TwitterSpamRFPred <- predict(RF_model, TwitterSpamTest)

# compare predicted outcome and true outcome
confusionMatrix(TwitterSpamRFPred, as.factor(TwitterSpamTest$label))


#4.	Use training dataset to train another machine learning model 
#   with the K Nearest neighbours algorithm.
Knn_model <- train(as.factor(label) ~ ., 
                data = TwitterSpamTrain, 
                method="kknn"
               )
Knn_model

#5. Use testing data to test and evaluate the model trained in step 4 and print the confusion matrix. 
# Predict the outcome on TwitterSapmTest
TwitterSpamTest <- TwitterSpam[-TwitterSpamSplit, ]
TwitterSpamKnnPred <- predict(Knn_model, TwitterSpamTest)
# compare predicted outcome and true outcome
confusionMatrix(TwitterSpamKnnPred, as.factor(TwitterSpamTest$label))


