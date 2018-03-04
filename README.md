# movie_recommendation
Data Source: https://grouplens.org/datasets/movielens/

The code is runnable in Apache SPARK. 

The system has been built in 3 steps. Step 1: Seperate dataset as taining, validation and testing set. Then tune some parameter of alternating least square algorithm. My choice of parameters to be tuned are number of features, number of iterations and lambda which is regularization parameter. These are done in process.py

In next step, the system has been built, verified and tested using k-fold cross validation. crossValidate.py is step 2 code.

In last step, the system generates ratings for a new user and outputs top 5 movie. recommend.py is our last step in this system.
