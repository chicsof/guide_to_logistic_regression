
install.packages("titanic")
library(titanic)
data("titanic_train")

#set our data
data <- titanic_train

# replace missing values for Age with mean
data$Age[is.na(Test$Age)] = mean(Test$Age, na.rm = TRUE)

# Remove attributes that are not valuable
install.packages("dplyr")
library(dplyr)
data <- data %>% select(-c(Cabin, PassengerId, Ticket, Name))

# use factors for the following attributes
for (i in c("Survived","Pclass","Sex","Embarked")){
    data[,i] = as.factor(data[,i])
}

# creates a genral linear model (fimily = binomial specifies logistic regression)
gml.fit = glm(Survived~., data = data, family = binomial(link ='logit'))
# we can see our estimated coefficients and associated p-values
summary(gml.fit)


# getting the R-squared
install.packages("DescTools")
library('DescTools')
# you will notice this is relatively low, it not easy to predict the servivol of a passanger only given a few details about them
# a lot of random factors are involved. There are also a lot of limitations in our model.
PseudoR2(gml.fit)

# We notice that this is a multivariate logistic regression, that takes the form of: lof(p(X)/(1-p(X)) = b0 + b1x1+ b2x2 + ... + bnxn

################################################## SOME LIMITATIONS YOU SHOULD CONSIDER #############################################

##### Confounding

# We often use logistic regression (and other models) to study the effects of various varibles on an the occurance of an event.
# We will use the following examples:
# > how does smoking affect the propability of a patient developing lung dicesses
# > how does the fact that an individuall is a student affect their propability of defulting in the bank account 

# When we conduct such a study meauring how many smokers have developed lung desses versurs how many non smokers, and how many
# students have defaulted versus how many non-students, we will most propably find a positive relationship. We cannot however, 
# blindly trust the bivarate analysis, I mean unleast we managed to get a perfectly random and balanced across all ages, sexes,
# incomes...sample there is a good chanche that our results are partially the result of other variables masked under the ones we
# have chosen to study. For example studies have shown that there are more older people that are smokers, also older people tend 
# to be more prone to desesses. So when we only use smoker/non smoker part of why our coeefifnt for that predictor whould be so 
# hight is because smokers are also older people. This is called bias due to confounding and occurs when confounders ( such us age)
# are not included in our model. If we take the example of students, when we only use student/non-student to predict the propability 
# of someone defolting we will find a possitive corelation. However, studies have shown that students with the same income as 
# non-students are less likely to default. Here the confounder is income, it just so happens that students will tend to have lower
# income, which makes them prone to defaulting. Therefore not including this in our models chould result in biased predictions. 

# In the older days, where creating such models was more deficult due to computing power and software not been as readily avalable
# people used stratifying in order to reduce the effects of confunding. They whould have to separete their data in subsets (strata)
# that were free of confounders. For example split by age groups or by student/non student. THen they whould conduct the estimation
# of coefficinets separetly for each group and use techniques such us pooling or weighted avaraging (if it made sence) to get an 
# overall estimation of the coefficients.

# if you are interested to learn more abou this you can wach this video from a Harvard lecture
# https://www.youtube.com/watch?v=hNZVFMVKVlc

# For reference, if we were to stricly define what a counfouder is, it whould be a variable that sutisfies the 
# following conditions:
# > Confunders has to be related to one of the significant predictors (e.g older people tend to be smokers more than younger)
# > It has to able be independantly related to the reaction (e.g. age is also on its own a significant predictor of somone developing
# a desease)
# It is not part of the causuall pathway (e.g smoking does not cause you to be old)

##### Multicolinearity
# Logistic regression also assumes no or little multicolinearity among the predictors. (see previous chapter)

##### Interaction terms
# Exacly the same as with linear regression (see previous chapter). Including the product (combination) of certain predictors
# chould optimize the model.
# log(P(x) \div (1-P(x))) = b0 + b1X1 + b2x2 + b3x1x2

##### Heteroscadasity (not relevant)
# In logistic regression, we excpect a propability to be systematically further away from the predicted y, in other words 
# homoscedasticity is not an assumption in this case. This is why we are unable to use least squares to find our best fitting line
# on the first place, since the residuals (distance of our points to the line) are reaching -/+ infinity.


############################################ EXAMPLE 2 ###########################################################

# Let's use onether example. Predicting whether or not a patient is diabetic.
# Our sample data set contains attributes such us BMI, age and measurments on glucose concentration, as well as a boolean that
# indicates whether or not each paitient is diabetic. We hope to use those attributes as predictors for the patiants diabetic 
# condiction. This time we will use onother technique for measuring how well our model preforms in giving acurate predictions.
# It is a common technique used in various models and not just the logistic regression. Basicly we will split the data that we
# have in two, the training data and the testig data. Our training data will be 80% of the total and it willl be used to estimate
# the coefficients and draw the S-curve by assigning propabilities (in other words training our model). The test data will be then
# fed to the trained model (without the attribute that gives out if a patiant is diabetic) and our model will give a prediction for
# each of the row. That prediction will be according to a threshold we have assigned, for example we may choose that if the propability
# of a patiant been diabetit is p >= 0.5 then we predict they are diabetic. Since we actually know if the patiants in the test data
# are diabetic or not, we can compare the actual to the predicted outcomes and get a % accuracy. We will see how this is done
# in details, as well as how we can optimize the chosen threshold for our specific problem.


#we can use a csv that is avalable online for our data, we can load this in R using the following code
data <- read.csv(file = "http://www2.compute.dtu.dk/courses/02819/pima.csv", head = TRUE, sep = ",")
############## SPLITING THE DATA#################################################################################################

# first we need to split our data, we do this with the use of caTools library (there are other ways too) ##
# the following creates an array of TRUE/FALSE values for every row in our data set,
# where TRUE is 80% of the data. It does this 'randomly', later on we will see techniques for spreading the success and failure
# rates proportionally across training and testing data
install.packages("caTools")
library(caTools)
split <- sample.split(data, SplitRatio =  0.8)
split
#transform to factors
data$type <- af.factor(data$type)
#we then split the data using that array, where TRUE is for the training and FALSE is for the testing
trainingData <- subset(data, split == TRUE)
testingData <- subset(data, split == FALSE)

#create the model using the TRAINING data
model <- glm(type ~., trainingData, family = binomial(link = 'logit'))
# review our model
summary(model)
#as expected BMI ang glu are significant (you can try and remove the predictors with a larger p-value but do this intrementally and note 
#the effect it has on the null and deviance residuals (see above) )

########lets visulize
#we will use only glu to plot our grapths easier
gml.fitGlu = glm(type ~ glu, data = trainingData, family = binomial(link ='logit'))
summary(gml.fitGlu) 
#first we will plot the best fiting line of propability of survivol over age 
#we need to get the propabilities for each row 
p <- predict(gml.fitGlu, trainingData, type = "response")
# get the lof of the odds, which is our y
y <- log(p/(1 - p))
# substitude with our calculated coefficients for the x axis
x <- -5.940285 + 0.042616  * trainingData$glu 
plot(x, y)
#Now let's transform it to the S-shaped curve with y from 0 to 1
#this is just the logistic function where i have substituted the coefficient for the intercept for our x with the ones
#calculates from our model summary
f <- function(x) {
  1/(1 + exp( -5.940285  + (0.042616  * x )))}
x <- trainingData$glu
#this scales the grapth
x <- -500 : 700
plot(x, f(x), type = "l")
#################

########################### CONFUSION MATRIX####################################################################################
# The predict() function allows us to feed new data in the trained model and receive a propability of each patiant on that data,
# been diabetic, we will use this on our testing data. 
# type = response outpouts the propability of each patiant been diabetic, we save the array of propabilities on pdata
pdata <- predict(model, testingData, type = "response")

# We will use those propabilities to predict whether the patiant is diabetic or not according to our model,
# given a threshold (p>0.55 in this case) and then compare the outcomes to the real values. We already know if those patiants
# are acually diabetic. We will compare the predicted to the actuall values using a confusion matrix.

#we will use the Caret library for that
install.packages("caret")
library(caret)


#create two arrays that contain TRUE or FALSE for each row (is diabatic TRUE, is not diabetic FALSE)
# one for the predicted data and one for the actuall, in order to compare them in R they need to be type factors (enumarations) and 
# have of the same levels (take on the same values)

#for the predicted TRUE is a row of propability > 0.55
predicted <- as.factor(pdata > 0.55)
#for the actual data, thether or not a patiant is diabetic is given by yes or no, so true is yes
real <- as.factor(testingData$type == "Yes")

# use caret to compute a confusion matrix
install.packages("e1071")
cm <- confusionMatrix(data = predicted, reference = real)
cm
# We can see that this has outputed a table decribing the following in each cell:

# predicted true that are actally true, also refered to as True Positive (TP)
# predicted true that are actually false, also refered to as False Positive (FP)
# predicted false that are actually false, also refered to as False Negative (FN)
# predicted false that are actually true, also refered to as True Negative (TN)

############### INTERPRETING ACCURACY
# It has also outputed some metrics on the performance of our model, lets explain them:

# > Sensitivity (also known as true positive, recall or hit rate) measures how well your model does in predicting positive values
# (predicting that the patient is diabet when they actually are): given by TP/(TP+FN)

# > Specificifity (also known as true negative rate) measures how well your model does in predicting negative values,
# (predicting that the patient is not diabet when they actually aren't): given by TN/(TN+FP)

# > Accuraccy gives an overall performance measurment on the accuracy of the results on a scale of 0 to 1, with 1 meaning
# that the model predicted everything correctly. It is given by: (TP + TN)/ (TP + FP + FN + TN)

# There is a big problem with using this metric as a performace indicator. Let's use an example where we create a model
# that simple always outputs FASLE. We use that model to predict whether or not an idividual has a very rare desease that is 
# estimated only 1 on 100 people have. If we use that model on a very large sample of people (that are already diagnosed 
# and know thether or not they have the desease) and then see how our model did, we will measure something like a 
# 0.99 acuracy. That is amazing right, our model is "99% accurate" ! This illustrates how this metric is biased by
# what proportions of possitive and negative values are avalable, a model chould do well simple by chance like we show in our
# example.

############ The kappa coefficient
# Cohen's Kappa coefficient is a mectric that tries to tackle the problem of bias in the measurment of accuraccy.
# It does so by 'eliminating' the chance of randomly getting a correct prediction from the equation.

# this 'chance' is just given by propability of randomly selecting a true value from the sample plus the propability of randomly
# selecting a false from the sample, since we want to measure both the chance of predicting TRUE when TRUE and FALSE when FALSE.
# In the sample (confusion matrix), we chould select a fasle value either from the total predicted false or the total actual false.
# So we need to multipy the propabilities assosciated with both. Similarly we chould select a positive value either from the total
# predicted possitive or the total acually possitive.

#from our confusion matrix table we can get:
# FN + FP
totalFALSE <- 47 + 15
# TP +TN
totalTRUE <-  5 + 7
# NT + NF 
totalNegative <- 47 + 5
# PT + PF
totalPositive <- 15 + 7 
# everything
totalSAMPLE <- 47 + 5 + 15 + 7

P_of_true_from_actuall = (totalTRUE/totalSAMPLE)
P_of_true_from_predicted = (totalPositive/totalSAMPLE)
P_of_false_from_actuall = (totalFALSE/totalSAMPLE)
P_of_false_from_predicted = (totalNegative/totalSAMPLE)

P_of_chance_for_TRUE <- P_of_true_from_actuall * P_of_true_from_predicted
P_of_chance_for_FALSE <- P_of_false_from_actuall * P_of_false_from_predicted
P_of_chance <- P_of_chance_for_TRUE + P_of_chance_for_FALSE
# We can see that our P_of_chance is very hight, this is because we have more samples of FALSE, there are more people that 
# are NOT diagnosed with diabities
P_of_chance

#to get the Kappa coefficient we whould use the fallowing formula...

# first we need to select accuracy from our confusion matric metrics
accuracy <- cm$overall['Accuracy'][[1]][1]
#Kappa is given by:
Kappa <- (accuracy - P_of_chance)/(1-P_of_chance)
#we can see this is the same value given by R in the confusion matrix
Kappa


########################################################################################################################

##################################### OPTIMIZING THE THRESHOLD #########################################################
#Choosing a better threshold
install.packages("ROCR")
library(ROCR)

resForTraining <-predict(model, trainingData, type = "response")
#choosing a value
ROCRPred = prediction(resForTraining,trainingData$type)
#measuring performance true and false predicted rate (0,1)
ROCRPref <-performance(ROCRPred, "tpr", "fpr")
plot(ROCRPref, colorize = TRUE)













