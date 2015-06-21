# Practical Machine Learning PA
21. Juni 2015  

### Introduction
The goal of this project is to predict human activity patterns from sensor attached to the body by using machine learning algorithms. The data of this project has been collected from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They performed barbell lifts correctly and incorrectly in 5 different ways. More information is available from the following website: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). The training and test data can be download here:

* https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
* https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


```
## Loading required package: lattice
## Loading required package: ggplot2
```

### Data cleanup data
The given training data contains about 20'000 observations in the test data set, and 20 observations in the final test data set for submission. The goal of the project is the construction of a predictor for the human activity stored in the variable `classe`. The five classes A, B, C, D and E correspondong to five activities sitting-down, standing-up, standing, walking, and sitting.

```r
data <- read.csv("data/pml-training.csv", na.strings = c("NA", "#DIV/0!",""))
summary(data$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

```r
# names(data)
```

The preduction should only depend on the motion sensor, and not on the user name and time stamp. Therefore the first seven columns are droped in the data set. The training observations are split in 864 windows of about 23 observations. Many columns contain statistical values about each window, e.g. the min, max, mean and standard deviation values. Therefore these columns contain many missing values. Since the activity is constant in each every time window, my first approach was to only make a prediction for every each time window. However this approach does not work, because the given test data does not contain complete time windows, only single observations. This means that we cannot make use of the time series structure of the data.


After understanding the structure of the given data, we split it into a training data set of 60%, and a testing data set for cross validation of 40%.

```r
set.seed(1234)
inTrain <- createDataPartition(data$classe, p=0.6, list=FALSE)
train <- data[inTrain,]
testing <- data[-inTrain,]
```

In the preprocessing we drop the first seven rows. Additionally we drop all columns with very small variance or many missing values. The remaing columns are all numeric and the creation of additional dummy variables for factors is not necessary.

```r
firstCols <- c(1:7)
nearZeroCols <- nearZeroVar(train)
numberNAs <- sapply(train, function(x) { sum(is.na(x))})
naCols <- which(numberNAs > 10)
badCols <- unique(c(firstCols,nearZeroCols,naCols))
```

### Decision/Classification tree
The simplest method to apply is the decision tree. We create the model, predict the outcome and then show the confusion matrix.

```r
treeModel <- train(classe~., method="rpart", data=train[,-badCols])
```

```
## Loading required package: rpart
```

```r
treePred <- predict(treeModel, testing[,-badCols])
confusionMatrix(treePred, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2029  638  644  567  209
##          B   44  505   49  232  211
##          C  155  375  675  487  383
##          D    0    0    0    0    0
##          E    4    0    0    0  639
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4904          
##                  95% CI : (0.4793, 0.5016)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3339          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9091  0.33267  0.49342   0.0000  0.44313
## Specificity            0.6334  0.91530  0.78388   1.0000  0.99938
## Pos Pred Value         0.4965  0.48511  0.32530      NaN  0.99378
## Neg Pred Value         0.9460  0.85114  0.87992   0.8361  0.88852
## Prevalence             0.2845  0.19347  0.17436   0.1639  0.18379
## Detection Rate         0.2586  0.06436  0.08603   0.0000  0.08144
## Detection Prevalence   0.5209  0.13268  0.26447   0.0000  0.08195
## Balanced Accuracy      0.7712  0.62399  0.63865   0.5000  0.72125
```

### Random forest
The random forest model takes a long time to run. I have to compare the results later.


### Logistic regression and PCA
This prediction with logistic regression algorithms could not be finished in time due to problems with missing values. Imputing missing values with the "knnImpute" did not help.


### Conclusions
Cross validation of the model has only be done by splitting the data into one training and test set. The confusion matrix shows clearly that the model is not very good and has a problem with the classification of class D. The model has an accuracy of 49% only, and a out of sample error of 51%.


