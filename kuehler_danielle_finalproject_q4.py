#Danielle Kuehler
#ITP 449 Summer 2020
#Final Project
#Q4

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.linear_model import Ridge
from yellowbrick.regressor import ResidualsPlot

#Question 4- Build a linear regression model

#Load data into dataframe using pandas
cars = pd.read_csv("auto-mpg.csv")
pd.set_option("display.max_columns", None)

#A.	Summarize the data set (describe). What is the mean of mpg?
print("Describe:\n",cars.describe())
print("\nA. The mean of MPG is 5.45")

#B.	What is the median value of mpg?
print("\nMedian:\n",cars.median())
print("\nB. The median value of MPG is 23.0")

#C.	Which value is higher – mean or median? What does this indicate in terms of the skewness of the attribute values? Make a plot to verify your answer.
print("\nC. Median is higher than mean, which means the data is skewed right")
plt.figure() #Create figure
plt.hist(cars['mpg']) #Create histogram
#Formatting
plt.xlabel("MPG")
plt.ylabel("Frequency")
plt.suptitle("C. Histogram of MPG\nKuehler_Danielle_FinalProject_Q4")
plt.show() #Display

#D.	Plot the pairplot matrix of all the relevant numeric attributes
sb.set_style("whitegrid") #Grid background
sb.pairplot(cars.iloc[:,1:5], kind="reg") #Include regression line to easier see correlation
plt.show() #Display

#E.	Based on the pairplot matrix, which two attributes seem to be most strongly linearly correlated?
#test = cars.drop(["No","model_year"], axis=1)
#c = test.corr().abs()
#s = c.unstack()
#so = s.sort_values()
print("\nE. Cylinders and Displacement seem most strongly linearly correlated")
#F.	Based on the pairplot matrix, which two attributes seem to be most weakly correlated.
print("\nF. Acceleration and weight seem most weakly linearly correlated")

#G.	Produce a scatterplot of the two attributes mpg and displacement with displacement on the x axis and mpg on the y axis.
plt.scatter(cars['displacement'], cars['mpg'], marker=".") #Scatter plot
#Formatting
plt.xlabel("Displacement")
plt.ylabel("MPG")
plt.suptitle("G. Scatterplot of MPG vs Displacement\nKuehler_Danielle_FinalProject_Q4")
plt.show() #Display

#H.	Build a linear regression model with mpg as the target and displacement as the predictor
x = cars["displacement"] #Predictor
y = cars["mpg"] #Target
X = x[:,np.newaxis] #All rows of cars dataset with empty column for prediction

#Fit training data to linear regression model
linReg = LinearRegression(fit_intercept=True)
linReg.fit(X,y) #Best fit
y_pred = linReg.predict(X) #Predict

#Ha. For your model, what is the value of the intercept β0 ?
print("\nHa. Intercept β0:", linReg.intercept_)

#Hb. For your model, what is the value of the coefficient β1 of the attribute displacement?
print("Hb. Coefficient β1:", linReg.coef_)

#Hc. What is the regression equation as per the model?
print("Hc. Regression equation: y=",linReg.coef_,"x + ", linReg.intercept_)
print("Hc. Regression equation: y = -0.06x + 35.17")

#Hd. For your model, does the predicted value for mpg increase or decrease as the displacement increases?
print("Hd. The predicted value for mpg decreases as the displacement increases")

#He. Given a car with a displacement value of 220, what would your model predict its mpg to be?
newframe = pd.DataFrame([[220]]) #New dataframe to predict
pred = linReg.predict(newframe)
print("He. Displacement is 220, MPG predicted: ", pred)

#Hf. Display a scatterplot of the actual mpg vs displacement and superimpose the linear regression line.
plt.scatter(X,y, marker=".") #Actual mpg vs displacement
#Formatting
plt.xlabel("Displacement")
plt.ylabel("Actual MPG")
plt.suptitle("Hf. Scatterplot and Regression of Actual MPG vs Displacement\nKuehler_Danielle_FinalProject_Q4")
plt.plot(X,y_pred) #Regression line
plt.show() #Display

#Hg. Plot the residuals
ridge = Ridge() #From sklearn
visualizer = ResidualsPlot(ridge)
visualizer.fit(X,y) #Fit to data
#Formatting
plt.suptitle("Hg. Plot of Residuals\nKuehler_Danielle_FinalProject_Q4")
visualizer.show() #Display

