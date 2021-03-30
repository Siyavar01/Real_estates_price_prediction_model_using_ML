# Real_estates_price_prediction_model_using_ML

Identification of Problem Domain

# •	Title

To develop an end-to-end Price prediction model for a real estate’s company using ML’s(Machine learning) algorithms and techniques. 


# •	Understanding the problem statement


	This model is being developed for a real estate’s company (we have assumed it to be Dragon real estate’s company).


	Currently, Dragon real estate’s company makes use of technicians who’re skilled in this concerned field.


	But, the accuracy with each they’re predicting the prices of housing is not satisfactory as there is approximately an error of 25% in their predictions.


	The management of the company is looking to maximize their profits and in order to do that they need to first minimize the percentage of error in their predictions.


	I’m going to come up with a ML model which is going to be trained on their given dataset. This model will be aimed to reduce error in the predictions and ultimately help in maximizing their profits.





 


# Knowledge of problem domain

•	What is Machine learning?

Arthur Samuel coined the term Machine Learning in the year 1959. He was a pioneer in Artificial Intelligence and computer gaming, and defined Machine Learning as “Field of study that gives computers the capability to learn without being explicitly programmed”.

•	How does Machine Learning works?

Machine Learning is, undoubtedly, one of the most exciting subsets of Artificial                   Intelligence. It completes the task of learning from data with specific inputs to the machine. It’s           important to understand what makes Machine Learning work and, thus, how it can be used in                    the future. 
The Machine Learning process starts with inputting training data into the selected algorithm. Training data being known or unknown data to develop the final Machine Learning algorithm. The type of training data input does impact the algorithm, and that concept will be covered further momentarily. 
To test whether this algorithm works correctly, new input data is fed into the Machine Learning algorithm. The prediction and results are then checked.
If the prediction is not as expected, the algorithm is re-trained multiple numbers of times until the desired output is found. This enables the Machine Learning algorithm to continually learn on its own and produce the most optimal answer that will gradually increase in accuracy over time.
•	Seven Steps of Machine Learning
	Gathering Data
	Preparing that data
	Choosing a model
	Training
	Evaluation
	Hyperparameter Tuning
	Prediction


# Knowledge of related problem

•	The most important aspect of this problem is our dataset i.e., “Boston_Housing.csv”.

•	The details about the dataset:

1. Title: Boston Housing Data

2. Sources:
   (a) Origin:  This dataset was taken from the StatLib library which is
                maintained at Carnegie Mellon University.
   (b) Creator:  Harrison, D. and Rubinfeld, D.L. 'Hedonic prices and the 
                 demand for clean air', J. Environ. Economics & Management,
                 vol.5, 81-102.

3. Past Usage:
   -   Used in Belsley, Kuh & Welsch, 'Regression diagnostics ...', Wiley, 
       1980.   N.B. Various transformations are used in the table on
       pages 244-261.
    -  Quinlan,R.. Combining Instance-Based and Model-Based Learning.
       In Proceedings on the Tenth International Conference of Machine 
       Learning, 236-243, University of Massachusetts, Amherst. Morgan
       Kaufmann.

4. Relevant Information:

   Concerns housing values in suburbs of Boston.

5. Number of Instances: 506

6. Number of Attributes: 13 continuous attributes (including "class"
                         attribute "MEDV"), 1 binary-valued attribute.

7. Attribute Information:

    1. CRIM      per capita crime rate by town

    2. ZN        proportion of residential land zoned for lots over 
                 25,000 sq.ft.

    3. INDUS     proportion of non-retail business acres per town

    4. CHAS      Charles River dummy variable (= 1 if tract bounds 
                 river; 0 otherwise)

    5. NOX       nitric oxides concentration (parts per 10 million)

    6. RM        average number of rooms per dwelling

    7. AGE       proportion of owner-occupied units built prior to 1940

    8. DIS       weighted distances to five Boston employment centres

    9. RAD       index of accessibility to radial highways

    10. TAX      full-value property-tax rate per $10,000

    11. PTRATIO  pupil-teacher ratio by town

    12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
                 by town

    13. LSTAT    % lower status of the population

    14. MEDV     Median value of owner-occupied homes in $1000's

8. Missing Attribute Values:  6.

•	The dataset “Boston_Housing.csv” with all of the values.



         

•	 Primarily, we’ve to answer these three questions before building our model:

	 Supervised, Unsupervised or Reinforcement learning?
	 Regression or Classification?
	 Batch learning or online learning techniques? 

•	Supervised, Unsupervised or Reinforcement learning?

•	After having a look at our dataset, we can clearly see that our dataset contains:
	   13 labels
	   1 target variable(“MEDV”)

•	Hence, we’re going to go with Supervised machine learning algorithms.

•	Regression or Classification?

•	The next question which arises is that whether this is a regression or classification problem.
•	In our dataset, the target variable(“MEDV”) consists of continuous set of values as it’s the price of housing and not categorical values (i.e. 0’s and 1’s).
•	Hence, Regression is the ideal type of solution for our problem.

•	Batch learning or online learning techniques?

•	Online learning techniques are not suitable for our dataset as it’s used in a scenario where there is a continuous supply of data.
•	Batch learning technique will be employed in our case where the model will be trained in batches.





 # Aptness of the programming Language

•	Why use Python for Machine learning?

	Simple and consistent

	Extensive Selection of libraries and frameworks

	Platform independence

	Great community and popularity


•	Software’s to be used:

	Python

	Jupyter Notebook

	Anaconda
