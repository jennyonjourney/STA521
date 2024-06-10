# Classification model for cloud detection with Spatial data


## Table of Contents
1. [Introduction](#Introduction)
2. [Project Statement](#Goals)
3. [Datasets](#Datasets)  
4. [Model](#Methodology)  
5. [Results](#Results)
6. [Conclusion](#Conclusion)
7. [Limitations/Implications](#Implication)


## 1.Introduction <a name="Introduction"></a>
The goal of this project is the exploration and modeling of cloud detection in the polar regions based on radiance recorded automatically by the MISR sensor abroad the NASA satellite Terra. This paper advises operational cloud detection algorithms that can deal with massive Multiangle Imaging SpectroRadiometer (MISR) provided by National Aeronautics and Space Administration (NASA)’s satellite without any human’s labeling. 


## 2.Project Statement <a name="Goals"></a>
Th major problem of this project is to build a better classifier in terms of accuracy based on the confusion matrix, F1 score, ROC and other metrics such as Matthews correlation coefficient (MCC) with spatial data. One key point is the proper visualization for spatial data and the other one is that each model has a unique assumption to be justified, and I will look into whehter those models justify models' unique assumptions.


## 3.Datasets <a name="Datasets"></a>
The data used in the paper were collected from 10 MISR orbits of path 26 over the Artic, northern Greenland, and Baffin Bay. The span of these orbits was approximately 144 days from April 28 through September 19, 2002, which is a daylight season in the Artic. Especially, Path 26 was chosen because it contains the richness of its surface features enough to train data for the algorithm to detect cloud aside from snow-covered and snow-free coastal mountains in Greenland. 

Overall, 57 data units with 7,114,248 1.1-km resolution pixels with 36 radiation measurements for each pixel were chosen to investigate after excluding 3 data units since their surfaces were open water, not appropriate for the study. The study concentrated on 275-m red radiation and repeated visits over time to improve the experts’ familiarity with the surface features for labeling process.

1. imagem1
   **Some features of the data:**
   Target Variable: *Default/Non-Default*
   Features: *11*
   Instances: *115,110*

2. imagem2
   **Some features of the data:**
   Target Variable: *Default/Non-Default*
   Features: *11*
   Instances: *115,229*

2. imagem3
   **Some features of the data:**
   Target Variable: *Default/Non-Default*
   Features: *11*
   Instances: *115,217*

### 3.1 Summary of data
The data sets, image1, image2, image3 have in total 345,556 data points with their x, y coordinates each. Based on the given study’s notation, the expert’s label 1 means clouds, -1 means no clouds and 0 means unlabeled. Each image has different proportion of clouds, 34.11% for image1, 17.77% for image2, and 18.44% for image3. In Table1, in total 23.43% of data points are clouds and 36.78% of data points are not clouds aside from 39.79% of unlabeled data points. It means that dataset’s classes are not evenly divided.

<img src="01_EDA/data_summary_table.png" width="250" />

**Is an i.i.d assumption for the samples justified for this dataset?**

When drawing maps, we can find that the images show different distribution of labels, which mean they clearly have different images showing different clouds location. According to three maps with x, y coordinates and expert labels, i.i.d assumption for the data sets is not justified for this data set because the maps show the clear division of 0, -1, +1 based on coordinates. It means that data points closed with each other in terms of x, y coordinates are especially dependent with each other. The interesting part of x, y coordinates of all images is that the range of x, y are almost the same with each other, for example, the range of x, y in image1 is 65 to 368 and 2 to 383 each whereas the range of x, y in image2 is 65 to 369 and 2 to 383 each. Based on this, we can assume that all images are about the same location (blocks) but different orbits. The paper focused on 10 different MISR orbits of path 26 out of 233 and each orbit represents one complete trip which normally took about 16 days. It implies that image1,2,3 represent the same location but different trips of MISR and time frame. This assumption again, means the data sets are not i.i.d.

![Picture2](https://github.com/jennyonjourney/predictive-modeling/01_EDA/map.png)

In order to understand the relationship of each variable, normalization is needed. I used the min-max normalization. When doing modeling even though it might not mandatory to normalize values for all modeling cases, but to get more precise, unbiased prediction, we need to normalize values of each variable. 

According to the correlation plots below, the relationship between expert labels and the individual features looks clear. The class 1 (cloud) has lower NDAI and the class -1 (not cloud) has higher NDAI. This is the same with SD and CORR variables but the boxplots show that both labels 1 and -1 have very long tails. Regarding SD, it seems that the class -1 (not cloud) has bigger variance. In Figure3.2, features themselves show a relatively strong correlation with each other. Especially, the radiance values of five different angles show a strong relationship with each other. The highest correlation among features is the relation between AF and AN.  

![Picture3](https://github.com/jennyonjourney/predictive-modeling/01_EDA/feature_comparison.png)


### 3.2 Preparation of data

**Split data**

Spatial autocorrelation is important factor to affect the model performance. Nearby pixels are more similar than distant ones and it can help to improve classification performance in terms of contextual spatial properties into the model but can also lead to overestimation.  In my paper, I conducted various of splitting data into train, validation, and test based on this assumption. In order to compare the model with different cross-validation samples, I used both random sampling method and sampling method for dependent data sets. Before splitting the data, I removed all unlabeled data points and 208,061 data points remained.

The split1 data sets are just a normal random sampling based on the assumption that data sets are i.i.d. The split2 data sets are proportionally split based on the cloud labels. In more detailed, the proportion of cloud label (+1) is 39% while the proportion of uncloud label (-1) is 61%. In the situation that it is necessary to build classification model given data sets, it would be better to split train, test, valid data sets by retaining the property and proportion of class of data sets.

The split3 data sets are the way of splitting image1, 2, 3 as just a train, valid, test set. Even though the hidden information of three images is not clear, for example which orbit and block of images it is, the purpose of model is to classify the clouds and non-clouds, the separation based on images themselves can be one of good options. The split4 data sets are the way of dividing data sets considering spatial autocorrelation. The data points closed with each other have a high probability to be dependent and show a similar classification result compared to the farther ones, it is needed to split train and test data which are not correlated with each other to improve the modeling precision and accuracy. 

Figure4 shows that we can divide the entire data into five pieces based on its x, y coordinates. Table1 shows the size of each split sets. Here I would like to recommend split set 3 and set 4 for not independent data sets

![Picture4](https://github.com/jennyonjourney/predictive-modeling/01_EDA/spatial_blocks.png)

![Picture5](https://github.com/jennyonjourney/predictive-modeling/01_EDA/size_split_data.png)


**Accuracy of trivial classifier**

In the scenarios that the data sets are imbalanced and concentrated on one specific label which a classifier has as its value, the high average accuracy is shown with a trivial classifier. Here the original data ('all data') in the table shows 61% of cloud-free proportion and this proportion is same with the split data based on the same proportional split of the original data sets. The reason that image split data set shows the highest accuracy is that the specific data sets designated as train and test sets have many proportions of cloud-free data by chance. This result suggests the baseline that classification modeling is not trivial but rather needs to consider both modeling parameters and cross-validation based on the right split of data sets.

![Picture6](https://github.com/jennyonjourney/predictive-modeling/01_EDA/acc_classifier.png)


**Feature selection**

Three of the best features are "NDAI", "SD" and "DF". I conducted importance test based on accuracy using varImp() function. Also, according to the correlation matrix, three camera angles out of six camera angles, AF, BF, AN have a relatively high correlation with other variables whereas CF and DF have relatively low correlation with other variables. So based on two aspects, two major features NDAI, SD and the radiation with DF angle can be selected as the best features for the model. 

![Picture7](https://github.com/jennyonjourney/predictive-modeling/01_EDA/feature_imp.png)


**CVmaster function for effective comparison**

For more effective modeling comparison, I created the CVmaster function which can be found in the zipfile with the name ‘CVmaster.R’ as a R script file.  Also, any relevant guideline to use this function is also included in the README file for reproducibility. Basically the function requires six different inputs like below: 1) classifier, 2) features, 3) label, 4) data, 5) k (number of cv fold), 6) loss. The classifier can be selected among six different models; Logistic Regression, LDA, QDA, KNN, Random Forest, Naïve Bayes. Please be careful to type the name of model because the name of model is very long. 



## 4.Model <a name="Methodology"></a>

### 4.1 Models

Using various packages, we tried to mitigate the bias in the baseline model. We rigorously assess these fairness tools to ensure their reliability and effectiveness, allowing us to offer informed recommendations on their practical use. We try various methods and metrics for each package to evaluate their effectiveness and compatibility with the client’s needs. 

**Logistic regression** 

AI Fairness 360(AIF 360) is an open-source library designed to help researchers detect, evaluate, and mitigate biases in machine learning algorithms. Depending on how they reduce bias, it offers a comprehensive suite of algorithms categorized as preprocessing, in-processing, and post-processing.

**LDA** 

AI Fairness 360(AIF 360) is an open-source library designed to help researchers detect, evaluate, and mitigate biases in machine learning algorithms. Depending on how they reduce bias, it offers a comprehensive suite of algorithms categorized as preprocessing, in-processing, and post-processing.

**QDA** 

AI Fairness 360(AIF 360) is an open-source library designed to help researchers detect, evaluate, and mitigate biases in machine learning algorithms. Depending on how they reduce bias, it offers a comprehensive suite of algorithms categorized as preprocessing, in-processing, and post-processing.

**Naive Bayes** 

AI Fairness 360(AIF 360) is an open-source library designed to help researchers detect, evaluate, and mitigate biases in machine learning algorithms. Depending on how they reduce bias, it offers a comprehensive suite of algorithms categorized as preprocessing, in-processing, and post-processing.

**Random Forest** 

AI Fairness 360(AIF 360) is an open-source library designed to help researchers detect, evaluate, and mitigate biases in machine learning algorithms. Depending on how they reduce bias, it offers a comprehensive suite of algorithms categorized as preprocessing, in-processing, and post-processing.

**KNN** 

AI Fairness 360(AIF 360) is an open-source library designed to help researchers detect, evaluate, and mitigate biases in machine learning algorithms. Depending on how they reduce bias, it offers a comprehensive suite of algorithms categorized as preprocessing, in-processing, and post-processing.


### 4.2 Metrics

**Accuracy & F1 score**
For assessing models’ fit, we can use various metrics. The most popular one is model’s accuracy. The accuracy of models can be calculated based on the confusion matrix with the actual labels of negative and positive and the predicted labels of negative and positive. The accuracy formula is the sum of True negative (TN) and True Positive (TP) divided by the whole data. The F-1 score is another way of assessing model performance. The formula of F-1 score is like follows; 2 x \[(Precision x Recall) / (Precision + Recall)\]. F-1 score (or F-measure) is a measure of a test’s accuracy. It can be also interpreted as harmonic mean of the precision and recall and a good metric showing how the model is effective. The higher F-1 score, we can interpret the better model and if the F-1 score is too much low, it means that the model is not that much effective.

Among six different models, Logistic regression, LDA, QDA, Naïve Bayes, Random Forest, KNN, in terms of accuracy, KNN and Random Forest have the highest values. Naïve Bayes also show the high accuracy but less than in F1-score, which means Naïve Bayes is less effective than KNN. Table4 shows that KNN looks a good classification model in terms of accuracy and F-1 score.


**ROC**
ROC curves are effective way to find the optimum threshold value. By increasing the threshold value, the model's sensitivity decreases, and specificity increases while the reverse happens if the threshold value is decreased. One should select the best threshold for the trade-off one wants to make. if you're more concerned with having a high specificity or low-false positive rate, pick the threshold that maximizes the true positive rate while keeping the false positive rate low. Firgure1 is one of the examples of ROC curve derived from random forest model. Depending on which part I can give up, specificity or sensitivity, we can decide the optimal cutoff.

**MCC**

One of additional metric to assess the model is Matthews correlation coefficient (MCC). It especially evaluates the effectiveness of the model when there is an imbalance between the two classes. The formula is MCC = (TP x TN - FP * FN) / √(TP+FP)(TP+FN)(TN+FP)(TN+FN). Here I used the logistic regression. The peak of the MCC occurs in the case where the specificity is somewhat larger than the sensitivity. One may wish to use a threshold either larger or smaller than the position of the peak of the MCC, depending on whether specificity or sensitivity is more highly valued.



## 5.Results <a name="Results"></a>

In empirically, LDA and Logistic regression have an assumption that the decision boundary is linear. So in this spatial classification, those two models are not appropriate. QDA assumption is the data should be generated from a normal distribution. The spatial data might be okay with quadratic decision boundaries but no there is no guarantee that data comes from normal distribution. Also, NativeBayes has an assumption that all predictors are independent. Since the spatial data points are dependent so this model is not appropriate. "KNN" is a good model with many sample sizes than predictor, but the choice of K is very important. KNN can still give poor results if the level of smoothness is not chosen correctly. Although all accuracy values of Train-Test data sets and all Test error rate of Test sets show Naive Bayes and Random Forest are good models, but KNN can be improved more with parameters and variable selections.

Figure8 shows the 20 observation sets’ test error rate box plots. Although the Random Forest model shows a good low error rates but we can find there are some outliers. On the other hand, KNN shows a relatively low test error rate and stable variance among observations sets. This model has to work well with various spatial data sets and in this perspective, KNN is the best classification model.

KNN has the property that choosing a right k (number of neighbors) is critically important in model performance. Figure9 shows the accuracy change according to different K. We can find that 23 is the optimal K number for the model’s performance. After changing the number of K into 23, the test error rate is not that much changed but the variance of test error rate among data sets is shrunken, and the overall box pot length is shorten.

**Pattern of misclassification**

When using KNN algorithm with K=23, the final mis-classification data points can be shown like below. Highly proportion seems to be misclassified. Many parts of misclassified data are in the left bottom side of regions. Since I run this KNN misclassification model based on ‘random’ train data sets, this mis-classification rate can be improved more with better cross-validation data sets.



## 6.Conclusion <a name="Conclusion"></a>
In order to make a good classification modeling, it is important that from the beginning, normalizing data, and setting the right number of data variables. KNN is the optimal model to classify results with a huge amount of data. Of course, there are lots of other machine learning models in the world including Neural network and Deep learning like CNN, but KNN is a quite good model to predict right classification in the spatial data sets. Especially when the data is not independent, KNN is a good option with a good choice of K and good cross validation sets of train, valid and test sets. 


## 7.Limitations/Implications <a name="Implication"></a>

It is clearly shown that it is important to determine the right value of parameter K (number of nearest neighbors). Furthermore, the method of calculating the distance for KNN modeling also matters and impacts the prediction ability. In practice side, one of the main KNN's disadvantages is slow speed with a large volume of data.
Also, based on 4(b), CNN model is highly recommended due to the spatial classification properties. Condensed nearest neighbor (CNN, the Hart algorithm) is an algorithm designed to reduce the data set for KNN classification. Also, Neural Network can be used as well. The plot below is the rough result of Neural Network classification. 

Documentation related to the project can be found at: 
[Documentation](https://github.com/jennyonjourney/predictive-modeling/05_documents)