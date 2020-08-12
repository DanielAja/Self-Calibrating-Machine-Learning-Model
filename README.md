<h1>Self-Calibrating-Machine-Learning-Model</h1>

<h4>Wells Fargo Campus Analytics Challenge 2020 <br />
Daniel Ajagbusi <br />
August 6th, 2020 <br />
University of Minnesota - Twin Cities </h4>

<h3>A visual description of the path of the data through:<h3>

<img alt="A visual description of the path of the data through" title="image" src="https://i.imgur.com/7okwyKK.png" /> <br />

<h2>DISCLAIMER:</h2>

_Though I refer to them as sets in the following passages, all “sets” other than omitted_data_col are actually python lists. This choice was made to allow the passages to feel more fluid in their conceptions._
---
<h3>1.) Data is provided in a spreadsheet</h3>
Because the data was provided in .xlsx format I decided to use the openpyxl python library to allow for simple read and write processes. However, upon reading the official rules I saw that we may have been expected to output the file as a .csv so I made note of this for later. After looking through the data I saw that each row was composed of 30 rational numbers and a letter, followed by a binary bit denoting the end result.
<h3>2.) Reading data into the program</h3>
I began by iterating over each row encoding the letter at the end of each row as its ASCII value minus 64, this is a form of ordinal encoding. (To result in A=1, B=2, … in order to make it easier to interpret for anyone who was viewing the data once entered into the program) This was based on the assumption that this model would only be used for data in the provided format.  The numbers did not need to be encoded, because machine learning processes work best with numbers. While this was happening the results (The final column) was also being recorded into its own set. These sets were made global to allow for interactions within other functions. 
<h3>3.) *Feature selection and correlation value calibrations ** </h3>
Because one of the main points of the challenge was to create a model with “the least set of feature variables, and no correlated variables in the set of predictors” I devised a system to self calibrate the data that would be used to train the model. This system works by incrementing the percent of top variables / minimum correlation allowed for features (columns) to be used and comparing the resulting output from the newly trained model of the training data to the known values of the training data. The process works by having a user specify an accuracy percentage then the function will find the smallest number of features needed to reach said accuracy. Because I approach this in a linear fashion it can be very time consuming if the precision parameter is set especially low. However, I don’t believe it would be too difficult to improve upon this in future iterations, because of the linear nature of logistic regression. But once a system is calibrated the values can be saved to be used instantly on the next run, hence why I set the variables of feature_variables_percentage_value = 50 and min_correlated_value_value = 5.64 respectively. Using 99% accuracy as a novel cut off point.
<h3>4.) Features deemed obsolete are omitted</h3>
Once we know the feature variable percentage and the minimum correlation value we would like to use to build our final model, through the process contained in the “run” function a set of omitted features are accumulated in the “omitted_data_col” set. These features are then removed from the working training data set.
<h3>5.) Training the model</h3>
After the undesirable features are removed from the training data set, this new data set in combination with the results data set acquired in step one are used to train the model. I made the choice to use logistic regression because of the binary nature of the results.
<h3>6.) Generating predictions</h3>
Reading in the data from the evaluation spreadsheet, similar to step one then removing the features denoted in step 4 and using the model trained in step 5, I was able to generate predictions and subsequently store said predictions of each of the 7000 rows denoted in the evaluation set provided. 
<h3>7.) Exporting the predicted data to a .xlsx file</h3>
Using the data generated in step 6 I cycled through the predictions along with a counter variable to export the data in the format denoted in the official rules. 
<h3>8.) *Creating a copy as a .csv file</h3>
As was denoted in step 1, I was unsure if we were expected to have our final results as a .xlsx or a .csv so I created a function to make a copy of the .xlsx generated in step 7, but export it as a .csv to ensure all rules were followed.
<h3>9.) Finding the F1 score</h3>
Because an additional part of the challenge was to calculate the F1 score. Using the sklearn.metrics library I compared the provided results set with a set created by having the model made in step 5, generate a result set using the training data. This resulted in an F1 score of 0.5614035087719299 

<h3>Summary:</h3> <br />
In short, I built a machine learning model that is able to calibrate itself to a user’s specified accuracy percentage, by excluding the lower half of the predetermined effective features and correlation values. This mode has the added benefit of never having to guess if 1, or 2 more features could have removed a large portion of errors, however, due to the linear nature of the calibration system the amount of time it takes is a heavy drawback. It is important to note that this is a one time cost if you choose to save the values it arrives at. Because the model I created has a F1 score > .5 mark (0.5614035087719299) I believe it is suitable for experimental use, however it may require some tweaking before it is ready for commercial use. With quickly self-calibrating machine learning models we may be able to determine relevant data and subsequently arrive at solutions much faster than we would have otherwise. Overall I believe that the process I began here can be utilized to help improve machine learning models already in use.


<h3>Environment Configuration:</h3>

`python==3.7.4(conda)`<br />
`numpy==1.14.5`<br />
`openpyxl==3.0.4`<br />
`pandas==0.23.1`<br />
`scikit-learn==0.22.1`<br />

<h3>Sources:</h3><br />
“1.13. Feature Selection¶.” Scikit, scikit-learn.org/stable/modules/feature_selection.html.<br />
Brownlee, Jason. “How to Choose a Feature Selection Method For Machine Learning.” Machine Learning Mastery, 30 June 2020, machinelearningmastery.com/feature-selection-with-real-and-categorical-data/.<br />
Brownlee, Jason. “Ordinal and One-Hot Encodings for Categorical Data.” Machine Learning Mastery, 29 June 2020, machinelearningmastery.com/one-hot-encoding-for-categorical-data/.<br />
Detective, The Data. “A Look into Feature Importance in Logistic Regression Models.” Medium, Towards Data Science, 14 Nov. 2019, towardsdatascience.com/a-look-into-feature-importance-in-logistic-regression-models-a4aa970f9b0f.<br />
Malik, Usman. “Applying Filter Methods in Python for Feature Selection.” Stack Abuse, Stack Abuse, stackabuse.com/applying-filter-methods-in-python-for-feature-selection/.<br />
Sunil RayI am a Business Analytics and Intelligence professional with deep experience in the Indian Insurance industry. I have worked for various multi-national Insurance companies in last 7 years. “Regression Techniques in Machine Learning.” Analytics Vidhya, 15 Apr.2020, analyticsvidhya.com/blog/2015/08/comprehensive-guide-regression/.<br />
Vickery, Rebecca. “Optimising a Machine Learning Model with the Confusion Matrix.” Medium, Towards Data Science, 27 Sept. 2019, towardsdatascience.com/understanding-the-confusion-matrix-and-its-business-applications-c4e8aaf37f42.<br />
