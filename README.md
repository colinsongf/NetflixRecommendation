# Netflix Recommendation
#### By Sol Vitkin
#### Last Update: 10/8/2015

OBJECTIVES: The purpose of this project is to fill in missing values for Netflix user-movie ratings. The Alternating Least Squares(ALS) algorithm is used to factor the user-movie matrix. Using this form of coordinate gradient descent, the algorithm converges upon a solution matrix that minimizes the difference between the original matrix values and the predicted factored matrix.

FILES: ALS Algorithm.py

RUNTIME INSTRUCTIONS: Putting two files, training.txt and testing.txt, training and testing data sets respectively, in the same directory as ALS Algorithm.py and running the script will create a new file with the missing values in the testing data filled in(pred_testingdata.csv). The RMSE for the training data is also printed.
