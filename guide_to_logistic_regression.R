################################################# LOGISTIC REGRESSION ##################################################
### similar to linear regression, however it is part of a group of models called classifiers (tries to group         ###
### elements/events) and this most often involves qualitative rather than quantitative data.                         ###
### Unlike linear regression, which is used mostly chosen for inference(studying the relationship between variables, ###
### effects of each other, strength and direction), we usually use logistic regression for predicting (or sometimes  ###
### studying) a binary outcome (has only two outcomes, patient has diabetes or not, transaction is fraud or legit).  ###
### The model does that by measuring the probability associated with each of the two and based on a chosen threshold ###
### will decide on one (e.g if p > 0.5 a transaction is fraud). It assigns those probabilities by taking into        ###
### account the predictors we have provided and their approximated coefficients as calculated from the data, that we ###
### have trained the model with. For example location of the transaction, time or money.                             ###
###                                                                                                                  ###
### As with linear regression, quantitative data is handled by assigning a dummy variable boolean (e.g isFraud = 1,  ###
### isNotFraud = 0). We are interested in finding how close a point is to 1 (in other words its probability of       ###
### success (fraud in this case) plotted on the y axis), as measured by the predictors (x, z...axis). The issue is   ###
### that we can get any value between 0 and 1, as well as values above and bellow that (e.g if the amount of money   ###
### used in a transaction is extremely large we might get something like p = 2.3 according to whatever coefficients  ###
### in y=a + bx +cx +..+e are calculated). This is not as useful and sometimes does not make sense, that's why we    ###
### use a transformation to fit our data exactly between 0 and 1.                                                    ###
###                                                                                                                  ###
### There are various transformation for that, in logistic regression we use the logistic function:                  ###
### P(x) =  e^(b0 + b1X)/ (1+ e^(b0 + b1X)), where b0, b1 are our coefficients, this makes an S-shaped curve where   ###
### the edges are 0 and one.                                                                                         ###
### We want to try and bring this in a more linear form, after manipulation we can see that:                         ###
### P(x)/(1-P(x)) = e^(b0 + b1X)                                                                                     ###
### P(x)/(1-P(x)), is called the the odds. It is the probability of success over the probability of failure. It is   ###
### used for example in horse-racing where if the odds are 1/3 the horse is likely to win one race but loose 3.      ###
### To make our equation more linear we can take the log. We notice that this is basically a linear equation using   ###
### the log of the odds on y-axis log( P(x)/(1-P(x))) = b0 + b1X, this makes things a lot easier (see more about why ###
### its helpful later on)                                                                                            ###
###                                                                                                                  ###
### Another issue is how we can find the 'best fit line', that best describes all our data points. In linear         ###
### regression we used the least squares approach. This required measuring the residuals (distance of our points to  ###
### the line) and trying to minimize that value by moving the line. However the distance of the points from a random ###
### line (close to the real) in linear regression is somehow constant, the error has a constant standard deviation.  ###
### In logistic regression when a predictor reaches extreme points the distance will tend to +/- infinity. For       ###
### example if we try and fit a line for the probability of a transaction been fraud based on the money involved; a  ###
### line close to the best fit would have very large distance on the points where the amount of money involved in a  ###
### transaction is large, or where it is very little. For that reason the method of maximum likelihood is preferred  ###
###                                                                                                                  ###
### We use the following steps to do that:

### 1:  Take  random strate line (a candidate 'best fit, just like in regression), that we think describes our points
### well.
### 2:  Use the P(x) =  e^(b0 + b1X)/ (1+ e^(b0 + b1X)) (given above) to transform that streat line to the S-shaped 
### curve that plots the propability from 0 to 1 for each of our x
### 3: We can now use those calculated p(x) foe each of our x to access how good this candidate line is. For each x 
### we can get the 'predicted' by the curve y (its P(x)), which in this case is also called the 'liklihood'
### 4: We then calculate the liklihood for ALL of the success cases (p(x)) and unsucessful cases (1-p(x)) (e.g
### propability, given by y value, of a fraud transaction acually been fraud, and propability, again as calculated by 
### our model of a legit transaction been not fraud). Whether a point is a seccuss of not, we know from the data we 
### have already colected. To get the total, we just need to multiply everything. This basicly, calculates the
### propability of us predicting everything correctly, so the higher that is the better. 
### 5: In reality we prefer to calculate the log of that propability instead. 'It just so huppens' that the log of
### the maximoum of the log of the likelihood is the same as the maximum of the likelihood.
### The reason why the log is favoured steams from the fact that it is easier to deferenciate a product. When looking
### for the max of an equation, we differenciate that equation and look for where it is equall to 0. If we take the 
### log of the equation we turn a difficult deferenciation of products to a addition (log(a*b)= log(a)+log(b) ), which
### is easy to solve. 
### 7: FInally, we choose the candidate line (which has our coefficients), that results in the higher log likelihood.
### (Software like R does this very very easilly using 'smart algorithms')










########################################################################################################################