# R_Studio_Analytics_TwitterSpam_3
[Twitter Spam 1](https://github.com/KianJay/R_Studio_Analytics_TwitterSpam) </br>
[Twitter Spam 2](https://github.com/KianJay/R_Studio_Analytics_TwitterSpam_2)</br>
R_Studio_Analytics_TwitterSpam_3</br>
<strong>Contributed by Kian(Duyoung Jang)</strong>

<h1>Assessment Introduction </h1>

We started to study supervised learning and to understand supervised algorithms. 
In this assessment,<br> you will apply supervised machine learning methods to classify Twitter spam using the provided dataset. Table 1 shows the features description of the dataset.<br>

<img src="https://user-images.githubusercontent.com/54985943/107938480-cf7ab580-6fc8-11eb-928f-f931b392a2a1.png" />
 
Twitter spam detection using R caret package
Follow instructions, complete all the tasks and organize your answers into an essay.<br> R script, R screenshot, your results and explanations should be covered for each question.



<h2>Questions</h2>
<h3>1.	Load dataset into R Studio, and randomly split the dataset to training dataset and testing dataset with the ratio of 8:2.</h3>
•	To load dataset into R studio: TwitterSpam <- read.csv("workspace/R/TwitterSpam/TwitterSpam.txt")<br>
•	To randomly split the dataset to training dataset and testing dataset with the ratio of 8:2set.seed(): set.seed(100)<br>
•	TwitterSpamSplit <- sample(1:nrow(TwitterSpam), 0.8 * nrow(TwitterSpam))<br>
•	set.seed() function sets the starting number that is used to generate a sequence of random numbers.<br>
•	TwitterSpamSplit, TwitterSpamTrain and TwitterSpamTest are set  the training and test with the ratio of 8:2<br>

<img  src="https://user-images.githubusercontent.com/54985943/107938489-d275a600-6fc8-11eb-8860-17fec5c05e75.png" />
<img  src="https://user-images.githubusercontent.com/54985943/107938500-d73a5a00-6fc8-11eb-8b22-d09c37b03c16.png" />

<h3>2.	Use training dataset to train a machine learning model with the random forest algorithm for Twitter spam classification </h3>
•	To use training dataset to train a machine learning model with the random forest algorithm for Twitter spam classification: I use “caret” library that is used for classification and Regression Training; Random Forest algorithms.<br>
•	library(caret) to Import caret library to train the dataset.<br>
•	RF_model <- train(as.factor(label) ~. ,  data = TwitterSpamTrain, method='ranger' )<br>
•	train() function is used to fit predictive models over different tuning parameters.<br>
•	Ranger as method used for Random Forest algorithm.<br>
•	RF_model to see the result which includes 1600 samples with mtry, splitrule, Accuracy and Kappa as tuning parameters.<br>
•	Mtry is the number of features used in the construction of each tree.<br>
•	Splitrule stands for splitting rule. For classification and probability estimation.<br>
•	Kappa is classification accuracy.<br>

<img src="https://user-images.githubusercontent.com/54985943/107938516-ddc8d180-6fc8-11eb-95e0-9432281d07fc.png" />
<img src="https://user-images.githubusercontent.com/54985943/107938522-e02b2b80-6fc8-11eb-9c59-16a7f570c346.png" />
<img src="https://user-images.githubusercontent.com/54985943/107938529-e1f4ef00-6fc8-11eb-9b5d-dc82217a3273.png" />

<h3>3.	Use testing dataset to test and evaluate the model trained in step 2 and print the confusion matrix.  </h3>
•	To use testing dataset to test and evaluate the model trained in step 2.<br> firstly, command : TwitterSpamRFPred <- predict(RF_model, TwitterSpamTest) to set the prediction variable, <br>
•	Predict function is to predict and is used for testing the data on an independent test.<br>
•	To print the confusion matrix. Predict the outcome on TwitterSpamTest.: confusionMatrix(TwitterSpamRFPred, as.factor(TwitterSpamTest$label)<br>
•	confusionMatrix functions is used to compare predicted outcome and true outcome. where I can find accuracy 0.97 and kappa 0.94.<br>

<img src="https://user-images.githubusercontent.com/54985943/107938538-e4efdf80-6fc8-11eb-9f02-d15821524ca1.png" />
<img src="https://user-images.githubusercontent.com/54985943/107938541-e6210c80-6fc8-11eb-86f7-82b99d2b82f0.png" />


<h3>4.	Use training dataset to train another machine learning model with the K Nearest neighbours algorithm.</h3>
•	To use training dataset to train another machine learning model with the K Nearest neighbours algorithm: Knn_model <- train(as.factor(label) ~ .,  data = TwitterSpamTrain, method="kknn"):<br>I use “caret” library that is used for classification and Regression Training; K nearest Neighbours algorithm<br>
•	Knn_model <- train(as.factor(label) ~. ,  data = TwitterSpamTrain, method=kknn )<br>
•	train() function is used to fit predictive models over different tuning parameters.<br>
•	kknn as method used for K-nearest neighbours algorithm that even ordinal and continuous variables can be predicted.<br>
•	Knn_model to see the result which includes 1600 samples with kmax, Accuracy and Kappa as tuning parameters.<br>
•	Kmax is maximum number of k, if ks is not specified.<br>
•	Kappa is classification accuracy.<br>

<img src="https://user-images.githubusercontent.com/54985943/107938603-fa650980-6fc8-11eb-96b4-ee1af1f3a8bb.png" />
<img src="https://user-images.githubusercontent.com/54985943/107938611-fc2ecd00-6fc8-11eb-84f4-ce9018d2a129.png" />

<h3>5.	Use testing data to test and evaluate the model trained in step 4 and print the confusion matrix.  </h3>
•	To use testing dataset to test and evaluate the model trained in step 4.<br> firstly, command : TwitterSpamKnnPred <- predict(Knn_model, TwitterSpamTest)<br>
•	Predict function is to predict and is used for testing the data on an independent test.<br>
•	To print the confusion matrix. Predict the outcome on TwitterSpamTest.: confusionMatrix(TwitterSpamKnnPred, as.factor(TwitterSpamTest$label)<br>
•	confusionMatrix functions is used to compare predicted outcome and true outcome. where I can find accuracy 0.91 and kappa 0.82.<br>

<img src="https://user-images.githubusercontent.com/54985943/107938626-00f38100-6fc9-11eb-9ab3-f5000f2b3d09.png" />
<img src="https://user-images.githubusercontent.com/54985943/107938632-02bd4480-6fc9-11eb-9a85-d05911fb3e65.png" />


<h3>6.	Comparing the performance of Twitter spam classifiers established in step 2 and step 4,<br> which algorithm can achieve better prediction results for this Twitter spam detection task? Why?</h3>


•	Comparing the performance of TwitterSpam classifiers established in step 2 Random Forest and step 4 K-Nearest Neighbours, I find some interesting information and results. Random Forest and K-nearest neighbours both are good algorithms to achieve prediction result. <br>
•	However, after comparing each other’s result and matrix, I found out that Random Forest model achieves better prediction results for this Twitter spam detection task.<br>
•	This is mainly because the random forest has higher accuracy and kappa. More specifically, the average of accuracy and kappa on the random forest are approximately 0.86 and 0.73, and on its confusion matrix and statistics, its accuracy is 0.97 and its kappa is 0.94.<br>
•	On the other hand, on k-nearest neighbours shows 0.78 accuracy and 0.57 kappa on its model, and on its confusion matrix and statistics it shows 0.91 and 0.82 each.<br>
•	Therefore, the Random Forest has better prediction results for this Twitter spam detection task with higher accuracy and kappa on model and confusion matrix.<br>

<h3> Conclusion </h3>
<img src="https://user-images.githubusercontent.com/54985943/107938666-0fda3380-6fc9-11eb-80a3-9ea6f8502dec.png" />

•	In conclusion, I implemented set.seed function to randomly split the dataset to training dataset and testing dataset with the ratio of 8:2 so I could use it to train the data with the random forest and k nearest neighbour algorithm for TwitterSpam classification using “caret” library. Also, I learned how to train the dataset using ranger method for random forest, and kknn method for k-nearest neighbour. In addition, I went through the confusion matrix for both random forest and k-nearest neighbour with confusionMatrix() functions. And finally I compared random forest with k-nearest neighbour to distinguish which algorithm can achieve better prediction results for this Twitter spam detection task and found out the Random Forest has better prediction results for this Twitter spam detection task with higher accuracy and kappa on model and confusion matrix.<br>
  
  <h3> Reference </h3>
Set.Seed function: 05.02.2021
https://livefreeordichotomize.com/2018/01/22/a-set-seed-ggplot2-adventure/<br>

A basic tutorial of caret: 03.02.2021
http://www.rebeccabarter.com/blog/2017-11-17-caret_tutorial/<br>

Caret Function & Method: 04.02.2021
https://topepo.github.io/caret/available-models.html<br>

K-Nearest Neighbours: 05.02.2021
https://www.rdocumentation.org/packages/kknn/versions/1.3.1/topics/kknn<br>
https://cran.r-project.org/web/packages/kknn/kknn.pdf<br>


