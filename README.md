# Diabetes_prediction

📌Heart Diseases Prediction📌
Heart Diseases

Image source

##   Introduction 
"Heart disease is broad term used for diseases and conditions affecting the heart and circulatory system. They are also referred as cardiovascular diseases. There are several different types and forms of heart diseases. The most common ones cause narrowing or blockage of the coronary arteries, malfunctioning in the valves of the heart, enlargement in the size of heart and several others leading to heart failure and heart attack." [Source]
 

Key facts according to WHO (World Health Organaizations)
Cardiovascular diseases (CVDs) are the leading cause of death globally.
An estimated 17.9 million people died from CVDs in 2019, representing 32% of all global deaths. Of these deaths, 85% were due to heart attack and stroke.
Over three quarters of CVD deaths take place in low- and middle-income countries.
Out of the 17 million premature deaths (under the age of 70) due to noncommunicable diseases in 2019, 38% were caused by CVDs.
Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol.
It is important to detect cardiovascular disease as early as possible so that management with counselling and medicines can begin.
Objectives
This notebook has two main objectives:

Explore the heart disease dataset using exploratory data analysis (EDA)
Exercise with classification algorithms for prediction (modelling)

##  Table of Contents
0. Introduction
1. Exploratory Data Analysis
1.0 Load Data
1.1 Data Dictionary
1.2 Data Pre-processing
1.3 Exploring Features
1.4 Correlations Heatmap
1.5 EDA Summary
2. Predictions
2.1 Scikit Learn Classifiers
2.2 Catboost, Lgbm and Xgboost
2.3 Model Explainablity
3. Concluding Remarks
4. References


## 1. Exploratory Data Analysis 
Libraries

import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from IPython.core.display import HTML
import matplotlib.pyplot as plt
from scipy.stats import uniform

import warnings
warnings.filterwarnings('ignore')

import os


## 1.0 Load Data 

df = pd.read_csv('/content/heart.csv')
df.head()

data.head() is a method that can be used to display the first few rows of a DataFrame or a Series object. It can be useful for quickly getting a sense of what the data looks like before proceeding with further analysis or processing.

 
data.dtypes
age           int64
sex           int64
cp            int64
trestbps      int64
chol          int64
fbs           int64
restecg       int64
thalach       int64
exang         int64
oldpeak     float64
slope         int64
ca            int64
thal          int64
target        int64
dtype: object
Note: From the data types we see that all features are int64/float64. But that is because some of the categorical features including the target (has disease/no disease) are already label encoded for us. We will, in the section below, see a detailed decreption of the features.

1.1 Data Dictionary 
age: age in years
sex: sex
1 = male
0 = female
cp: chest pain type
Value 0: typical angina
Value 1: atypical angina
Value 2: non-anginal pain
Value 3: asymptomatic
trestbps: resting blood pressure (in mm Hg on admission to the hospital)
chol: serum cholestoral in mg/dl
fbs: (fasting blood sugar > 120 mg/dl)
1 = true;
0 = false
restecg: resting electrocardiographic results
Value 0: normal
Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
thalach: maximum heart rate achieved
exang: exercise induced angina
1 = yes
0 = no
oldpeak = ST depression induced by exercise relative to rest
slope: the slope of the peak exercise ST segment
Value 0: upsloping
Value 1: flat
Value 2: downsloping
ca: number of major vessels (0-3) colored by flourosopy
thal:
0 = error (in the original dataset 0 maps to NaN's)
1 = fixed defect
2 = normal
3 = reversable defect
target (the lable):
0 = no disease,
1 = disease
Note on the target label:

Diagnosis of heart disease (angiographic disease status)
Value 0: < 50% diameter narrowing
Value 1: > 50% diameter narrowing

Notes from the discussion forum of the dataset:

data #93, 159, 164, 165 and 252 have ca=4 which is incorrect. In the original Cleveland dataset they are NaNs.
data #49 and 282 have thal = 0, also incorrect. They are also NaNs in the original dataset.
Action: Drop the faulty data! (7 data entry will be dropped)

Back to top

1.2 Data pre-processing 
1.2.1 Drop faulty data
Based on our investigation we did above, we will drop 7 rows.

The length of the data now is 296 out of 303!
1.2.2 Rename columns for the sake of clarity
The feature names in the dataset are abbreviated and hard to understand their meaning. A full medical/technical name is hard enough to understand for most of us let alone their short form. So to make them a little bit easier to read we will, here under, change the column names of the data frame using information from the UCL data repository.
We'll also replace the coded categories (0, 1, 2,..) to their medical meaning ('atypical angina', 'typical angina', etc. for example)
Ref: I borrowed Rob Harrand's idea of re-naming the columns.
data.dtypes
age                            int64
sex                           object
chest_pain_type               object
resting_blood_pressure         int64
cholesterol                    int64
fasting_blood_sugar           object
resting_electrocardiogram     object
max_heart_rate_achieved        int64
exercise_induced_angina       object
st_depression                float64
st_slope                      object
num_major_vessels              int64
thalassemia                   object
target                         int64
dtype: object
data.head()
age	sex	chest_pain_type	resting_blood_pressure	cholesterol	fasting_blood_sugar	resting_electrocardiogram	max_heart_rate_achieved	exercise_induced_angina	st_depression	st_slope	num_major_vessels	thalassemia	target
0	63	male	asymptomatic	145	233	greater than 120mg/ml	normal	150	no	2.3	upsloping	0	fixed defect	1
1	37	male	non-anginal pain	130	250	lower than 120mg/ml	ST-T wave abnormality	187	no	3.5	upsloping	0	normal	1
2	41	female	atypical angina	130	204	lower than 120mg/ml	normal	172	no	1.4	downsloping	0	normal	1
3	56	male	atypical angina	120	236	lower than 120mg/ml	ST-T wave abnormality	178	no	0.8	downsloping	0	normal	1
4	57	female	typical angina	120	354	lower than 120mg/ml	ST-T wave abnormality	163	yes	0.6	downsloping	0	normal	1
1.2.3 Grouping Features (by data type)
As we have seen above there are three datatypes i.e object, int and floats. Let's group them according to type.
1.3 Exploring Features/Target 
In this section we'll investigate all the features (including the target) in detail. We will look at the statistical summary when possible and the distributions of some of them as well, starting from the target.

1.3.1 Target distribution
We observe that the target is fairly balanced with ~46% with no heart disease and ~54% with heart disease. So no need to worry about target imbalance.


1.3.2 Numerical Features
Statistical summary
For the numerical features we can apply the handy pandas data.describe() method and get the global statistical summary.

data[num_feats].describe().T
count	mean	std	min	25%	50%	75%	max
age	296.0	54.523649	9.059471	29.0	48.0	56.0	61.00	77.0
cholesterol	296.0	247.155405	51.977011	126.0	211.0	242.5	275.25	564.0
resting_blood_pressure	296.0	131.604730	17.726620	94.0	120.0	130.0	140.00	200.0
max_heart_rate_achieved	296.0	149.560811	22.970792	71.0	133.0	152.5	166.00	202.0
st_depression	296.0	1.059122	1.166474	0.0	0.0	0.8	1.65	6.2
num_major_vessels	296.0	0.679054	0.939726	0.0	0.0	0.0	1.00	3.0
Statistical summary of the numerical features
Age :
The average age in the dataset is 54.5 years
The oldest is 77 years, whereas the youngest is 29 years old
Cholesterol:
The average registered cholestrol level is 247.15
Maximum level is 564 and the minimum level is 126.
Note: According to [6], a healthy cholesterol level is 
 and usually high level of cholesterol is associated with heart disease.
Resting blood pressure:
131 mean, 200 max and 94 min
Max heart rate achieved:
The abverage max heart rate registered is 149.5 bpm. The Maximum and the minumum are 202 and 71bpm respectively.
St_depression:
The average value of st_dpression is 1.06. Max is 6.2 and the minimum is 0.
Number of major blood vessels:
A maximum of 3 and a minimum of 0 major blood vessels are observed. The mean value is 0.68.
Back to top

Distribution: Density plots

Pair-plots

Selected Features
Below are reg-plots of some selected features showing the linear relation with Age, similar to the first column in the pair-plot above. We observe that:

Except maximum_heart_rate_achieved, the others are positively and linearly related with age (albeit a weaker relation with st_depression).
Younger patients with higher maximum_heart_rate_achieved are more likely to have a heart condition.
Lower st_depression regardless of age is also likely an indication of a heart disease.

1.3.3 Categorical Features
We use a count plot to visualize the different categories with respect to the target variable. Two things we could take note of are the distribution of each category in the dataset and their contribution to the probability of correct prediction of the target variable, i.e has disease (=1) or has no disease (=0). Below is the summary of the categorical features.

Chest Pain:
More than 75% of the patients experience either typical angina or non-angina chest pain.
Patients who experienced atypical angina or non-angina chest pain are more likely to have a heart disease.
Resting Electrocardiogram:
Patients with Left ventricular hypertrophy are the fewest (~1.4%). The rest is almost a 50-50 split between patients with ST-T abnormality and those with normal REC tests.
ST-T abnormality seem to have a better correlation with the target, i.e the majority of patients with this kind of REC test ended up with a heart disease.
ST-Slope:
Most patients have a downsloping or flat ST-Slope of their REC test.
downsloping ST-Slopes are a strong indication that a patient might have a heart disease.
Thalassemia:
Most patients have a normal or reversable defect
Patients who have thalassemia defects (reversable + fixed) are less likely to have a heart disease. Whereas, those with normal thalassemia are more likely to have a heart condition. Sounds not intuitive.
Fasting blood sugar
Patients with lower (less than 120mg/ml) fasting blood sugar are the majority in our dataset consisting of ~85% of the sample.
Having lower resting blood sugar tends to increase the chances (~54%) of a heart disease.
Exercise Induced Angina
Two-third of the patients showed no exercise induced angina.
76% of the patients with exercise induced angina had no heart conditions. Whereas ~69% of the patients who did not experience exercise induced angina were diagnosed with heart condition.
Sex
More patients in the sample data are male.
Females seem to suffer from heart condition more than males.
Distribution: Count plots

1.4 Correlation Heatmaps 
Correlation heatmap is a useful tool to graphyically represent how two features are related to eachother. Depending upon the data types of the features, we need to use the appropriate correlation coefficient calculation methods. Examples are pearson's correlation coefficient, point biserial correlation, cramers'V correlation and etc.

1.4.1 Pearson's correlation
The Pearson correlation coefficient ― is a measure of linear correlation between two sets of data. It is the ratio between the covariance of two variables and the product of their standard deviations; thus it is essentially a normalised measurement of the covariance, such that the result always has a value between −1 and 1. (ref. )

1.4.2 Point biserial correlation
A point-biserial correlation is used to measure the strength and direction of the association that exists between one continuous variable and one dichotomous variable. It is a special case of the Pearson’s product-moment correlation, which is applied when you have two continuous variables, whereas in this case one of the variables is measured on a dichotomous scale [ref. ].

1.4.3 Cramer's V correlation
In statistics, Cramér's V is a measure of association between two nominal variables, giving a value between 0 and +1 (inclusive). It is based on Pearson's chi-squared statistic and was published by Harald Cramér in 1946. [ref. ]

Back to top

1.5 EDA Summary: 
Data size: 303 rows and 14 columns (13 independent + one target variable) > later reduced to 296 after removing faulty data points!
Data has no missing values
Features (columns) data type:
Six features are numerical
The rest (seven features) are categorical variables
Target variable is fairly balanced, 54% no-disease to 46% has-disease
Correlations:
Correlation between features is weak at best
From the numerical features num_major_vessels, max_heart_rate_achieved and st_depression are reasonabily fairly correlated with the target variable at -0.47, 0.43 and -0.43 correlation coefficient respectively.
From the categorical features chest_pain_type, num_major_vessels, thalassemia, and exercise_induced_angina are better correlated with the target variable, thalassemia being the highest at 0.52.
Cholestrol (to my surprize, but what do I know?) has less correlation with heart desease.
Takeaway: features that have higher predictive power could be, chest_pain_type, num_major_vessels, thalassemia, exercise_induced_angina max_heart_rate_achieved and st_depression. We will see which features will appear as imporatnt by the classification models.

2. Predictions 
Note : We have only 297 case (after data cleaning) which is a very small amount of data to do any serious prediction. Therefore, any conclusion made must be taken with cautions. This notebook is merely an excercise on binary classification algorithms.

2.1 Scikit Learn Classifiers 
This is a binary classification problem (has-disease or no-disease cases). Scikit learn offers a wide range of classification algorithms and is often the starting point in most/traditional machine learning challenges, so we start by exploring few of the classification alorithms from the sklearn libarary such as Logistic Regression, Nearest Neighbors, Support Vectors, Nu SVC, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, Naive Bayes, Linear Discriminant Analysis, Quadratic Discriminant Analysis and Neural Net. Let's first build simple models using the above mentioned ML algorithms and later we will optimize them by tuning the parameters.

2.1.1 Performance metric
There are several metrics that can be used to gauge the performance of a given classification algorithm. The choice of the 'appropriate' metrics is then dependent on the type of problem we are dealing with. There are case where, for example, accuracy can be the right choice and in some other case a recall or precision could be more fitting to the purpose. Since we are dealing with medical case (classify if a case is positive for heart disease or not), we could use recall (true positive rate or sensitivity) as performance metrics to choose our classifier. Note here that we do not want to classify positive (has disease) cases as negative (no disease).

Confusion matrix : A confusion matrix (aka an error matrix) is a specific table layout that allows visualization of the performance of a supervised learning algorithm. Each row of the matrix represents the instances in an actual class while each column represents the instances in a predicted class [wiki]. The table below is an example of a confusion matrix for a binary classification from which other terminologies/metric can be derived. Some of the metrics are described below.


Image credit >>

Key:

Term	Meaning	Descriptions
TP	True Positive	Positive cases which are predicted as positive
FP	False Positive	Negative cases which are predicted as positive
TN	True Negative	Negative cases which are predicted as negative
FN	False Negative	Positive casea which are predicted as negative
Accuracy : Measures how many of the cases are correctly identified/predicted by the model, i.e correct prediction divided by the total sample size.


Recall: Measures the rate of true positives, i.e how many of the actual positive cases are identified/predicted as positive by the model.


Precision: Measures how many of the positive predicted cases are actually positive.


F1-Score : Combines the precision and recall of the model and it is defined as the harmonic mean of the model’s precision and recall.


ROC curves : A receiver operating characteristic (ROC) curve, is a graphical plot which illustrates the performance of a binary classification algorithm as a function of ture positive rate and false positive rate.

Back to top

2.1.2 Performance metrics summary table
 	Classifier	Accuracy	ROC_AUC	Recall	Precision	F1
0	Logistic Regression	86.490000	0.920000	0.910000	0.820000	0.860000
9	Linear DA	85.140000	0.920000	0.890000	0.820000	0.850000
10	Quadratic DA	85.140000	0.900000	0.830000	0.850000	0.840000
5	Random Forest	83.780000	0.920000	0.830000	0.830000	0.830000
4	Decision Tree	82.430000	0.820000	0.830000	0.810000	0.820000
6	AdaBoost	82.430000	0.860000	0.910000	0.760000	0.830000
7	Gradient Boosting	82.430000	0.900000	0.890000	0.780000	0.830000
8	Naive Bayes	82.430000	0.920000	0.860000	0.790000	0.820000
3	Nu SVC	81.080000	0.910000	0.910000	0.740000	0.820000
11	Neural Net	78.380000	0.880000	0.940000	0.700000	0.800000
2	Support Vectors	64.860000	0.800000	0.890000	0.580000	0.700000
1	Nearest Neighbors	55.410000	0.600000	0.310000	0.550000	0.400000
2.1.3 ROC curves

2.1.4 Confusion matrix

Now we have seen all the performance metrics of the classifiers, it is decision time for us to choose the best possible classifier algorithm. Based on precision LR ranks first (86%); whereas if we see the recall, Neural Nets ranks first with 94%. In the case of precision, QDA ranks first with 85%. So which one to choose? The F1-score can give us a balance between recall and precision. LR happens to have the best F1-score so we choose Logistic Regression as our best classifier.

Note: If I were consulting a clinic doing a heart disease screening test, I would like to strike a perfect balance between precision and recall (I don't want the clinic to risk their reputation of by handing out too many false positive result but all without risking their clients' health by predicting too many false negatives). Therefore, I would advice them to choose the model which gives a higher F1-score, i.e the Logistic regression model.

2.1.5 Parameter Tuning (RandomizedSearch): LogisticRegression
So chosen our best classifier, the Logistic regression model. However, this was achieved with default parameters. The intuition is that we could further improve our model with tuned parameters. Let's see if could achieve that using the scikit-learn RandomizedSearch algorithm.

Best Hyperparameters: {'C': 0.2835648865872159, 'penalty': 'l2', 'solver': 'liblinear'}
              precision    recall  f1-score   support

           0       0.91      0.82      0.86        39
           1       0.82      0.91      0.86        35

    accuracy                           0.86        74
   macro avg       0.87      0.87      0.86        74
weighted avg       0.87      0.86      0.86        74

plot_confusion_matrix(lr, X_val, y_val)
<sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7da443589b10>

Remark : It turns out that our base model (default params) is not bad at all. Parameter tuning did not help to further increase the performance.

Back to top

2.2 Catboost, Lgbm and Xgboost 
In the above section (&&2.1) we have seen classifiers out of the scikit-learn library. Now we will try the modern (boosted trees) ML algorithms such as the catboost, xgboost and lgbm. They are optimized machine learning algorithms based on the gradient-boosting technique. Depending on the problem at hand, one algorithm is may be better suited than others. For detailed info one can easily refer to their documentations.

2.2.1 Performance metrics summary table
 	Classifier	Accuracy	ROC_AUC	Recall	Precision	F1
0	Catboost	83.780000	0.920000	0.860000	0.810000	0.830000
2	light GBM	82.430000	0.910000	0.860000	0.790000	0.820000
1	xgbbost	79.730000	0.920000	0.830000	0.760000	0.790000
2.2.2 Confusion matrix

Remark : Here we can see that the lgbm calssifier is marginally better than the other two and we will go for it. Following the same procedure, we will try to tune the parameters in the next section.

2.2.3 Parameter Tuning (RandomizedSearch): LGBMClassifier
              precision    recall  f1-score   support

           0       0.94      0.79      0.86        39
           1       0.80      0.94      0.87        35

    accuracy                           0.86        74
   macro avg       0.87      0.87      0.86        74
weighted avg       0.88      0.86      0.86        74

plot_confusion_matrix(lgbm, X_val, y_val);

Remark: In this case (Lgbm) hyper-parameter tuning gave better results than the base model. We have increased the recall value from 86% to 94%. Which means we have decrease the rate of false negatives from 5 cases to 2 in our validation set and we have also decreased the false positive cases by 1. Marginal but we will take every percentage point we can get.

Back to top

import joblib

# Train the LGBMClassifier model and evaluate it on the validation set
lgbm = LGBMClassifier(**params)
lgbm.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
y_pred = lgbm.predict(X_val)
print(classification_report(y_val, y_pred))

# Save the trained model using joblib
joblib.dump(lgbm, 'Heart_Disease_Prediction.pkl')
[LightGBM] [Warning] min_data_in_leaf is set=80, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=80
              precision    recall  f1-score   support

           0       0.94      0.79      0.86        39
           1       0.80      0.94      0.87        35

    accuracy                           0.86        74
   macro avg       0.87      0.87      0.86        74
weighted avg       0.88      0.86      0.86        74

['Heart_Disease_Prediction.pkl']
2.3 Model Explainablity 
One of the challenges of a machine leaning project is explaining the model's prediction. A model might consider some features more important than other for its prediction. Another model might weigh other features as more important. Permutation importance and SHAP are two methods one can use to understand which features were selected to have the most impact on our model's prediction.

2.3.1 Permutation importance:
The permutation importance is defined to be the decrease in a model score when a single feature value is randomly shuffled. The procedure breaks the relationship between the feature and the target, thus the drop in the model score is indicative of how much the model depends on the feature [3]. In other words, permutation importance tell us what features have the biggest impact on our model predictions.

import eli5
from eli5.sklearn import PermutationImportance

perm_imp = PermutationImportance(lgbm, random_state=seed).fit(X_train, y_train)
eli5.show_weights(perm_imp, feature_names = X_val.columns.tolist())
Weight	Feature
0.0901 ± 0.0127	num_major_vessels
0.0730 ± 0.0276	chest_pain_type
0.0288 ± 0.0340	st_slope
0.0099 ± 0.0175	max_heart_rate_achieved
0.0054 ± 0.0067	st_depression
0 ± 0.0000	thalassemia
0 ± 0.0000	exercise_induced_angina
0 ± 0.0000	resting_electrocardiogram
0 ± 0.0000	fasting_blood_sugar
0 ± 0.0000	resting_blood_pressure
0 ± 0.0000	sex
-0.0009 ± 0.0144	cholesterol
-0.0045 ± 0.0114	age
2.3.2 SHAP:
SHAP, a short name for SHapely Additive ExPlanations, is method used to explain the output of a machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions [5]. SHAP has a rich functionality (methods) by which we can visualize/interpret the output of our models. Below we use the shap.summary_plot() to identify the impact of each feature has on the predicted output.

import shap
shap.initjs()
explainer = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val, feature_names=features, plot_type="bar")


shap.summary_plot(shap_values[1], X_val)

3. Concluding Remarks 
At the start of this notebook, we've laid out what we wanted to do with this project; to explore the heart disease dataset (EDA) and practice binary classification (modelling). In part one (EDA) we did explore the dataset, did sanity check and removed some 'faulty' data and other pre-processing. We also tried to identify correlation between features and also with the target variable. In part two we practiced how to set-up binary classifiers; first starting with base models and finally arriving at our best model via hyper-parameter tuning. Some of the highlights are summarized below.

Our best model happens to be lgbm classifier (tuned with randomizedSearch)
According to both eli5 permutation importance and SHAP the three most important features of the model are num_major_vessels, chest_pain_type, and st_slope. These features are also among better correlated features from our EDA.
Contrary to my intuition cholesterol happens to be not an important feature for the model (both eli5 and SHAP did not pick this feature as important).
Although is not shown in this notebook, varying the test/train ratio resulted different performance metrics for the classifiers we have on out list. So if you change the ratio you might get different results.
