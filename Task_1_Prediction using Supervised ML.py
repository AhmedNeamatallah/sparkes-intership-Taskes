"""
Author ==> Ahmed Mohamed Neamatallah
TASK 1 ==> Prediction using Supervised ML
                To Predict the percentage of marks of the students based on the number of hours they studied
"""

# importing the required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Reading the DataFrame
df = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
df.info()

#visualize our Data
sns.set_style('darkgrid')
sns.scatterplot(y= df['Scores'], x = df['Hours'])
plt.title('Marks Vs Study Hours', size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()

#From the above scatter plot there looks to be correlation between the 'Marks Percentage' and 'Hours Studied',
# Lets plot a regression line to confirm the correlation.

sns.regplot(x= df['Hours'], y= df['Scores'])
plt.title('Regression Plot',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()
print(df.corr())

"""
Splitting the Data
"""
# Defining X and y from the Data
X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

# Spliting the Data in two
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

"""
 Fitting the Data into the model
"""
regression = LinearRegression()
regression.fit(train_X, train_y)
print("---------Model Trained---------")

"""
Predicting the Percentage of Marks
"""
pred_y = regression.predict(val_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in val_X], 'Predicted Marks': [k for k in pred_y]})


"""
Comparing the Predicted Marks with the Actual Marks
"""
compare_scores = pd.DataFrame({'Actual Marks': val_y, 'Predicted Marks': pred_y})

#Visually Comparing the Predicted Marks with the Actual Marks

plt.scatter(x=val_X, y=val_y, color='blue')
plt.plot(val_X, pred_y, color='Black')
plt.title('Actual vs Predicted', size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()

"""
Evaluating the Model
"""
# Calculating the accuracy of the model
print('Mean absolute error: ',mean_absolute_error(val_y,pred_y))

"""
What will be the predicted score of a student if a student studies for 9.25 hrs/ day?
"""
hours = [9.25]
answer = regression.predict([hours])
print("Score = {}".format(round(answer[0],3)))


'''
ANSWER ==> According to the regression model if a student studies for 9.25 hours a day he/she is likely to score 93.893 marks
'''
