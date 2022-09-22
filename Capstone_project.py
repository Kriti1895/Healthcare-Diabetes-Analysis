#!/usr/bin/env python
# coding: utf-8

# In[122]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[123]:


data=pd.read_csv("C:/Users/hp/Downloads/Project_2/Project 2/Healthcare - Diabetes/health care diabetes.csv")


# In[124]:


data.head()


# In[125]:


data.isnull().any()


# In[126]:


data.info()


# In[127]:


Positive = data[data['Outcome']==1]
Positive.head(5)


# In[128]:


data['Glucose'].value_counts().head(7)


# In[129]:


plt.hist(data['Glucose'])


# In[130]:


data['BloodPressure'].value_counts().head(7)


# In[131]:


plt.hist(data['BloodPressure'])


# In[132]:


data['SkinThickness'].value_counts().head(7)


# In[133]:


plt.hist(data['SkinThickness'])


# In[134]:


data['Insulin'].value_counts().head(7)


# In[135]:


plt.hist(data['Insulin'])


# In[136]:


data['BMI'].value_counts().head(7)


# In[137]:


plt.hist(data['BMI'])


# In[138]:


data.describe().transpose()


# In[139]:


plt.hist(Positive['BMI'],histtype='stepfilled',bins=20)


# In[140]:


Positive['BMI'].value_counts().head(7)


# In[141]:


plt.hist(Positive['Glucose'],histtype='stepfilled',bins=20)


# In[142]:


Positive['Glucose'].value_counts().head(7)


# In[143]:


plt.hist(Positive['BloodPressure'],histtype='stepfilled',bins=20)


# In[144]:


Positive['BloodPressure'].value_counts().head(7)


# In[145]:


plt.hist(Positive['SkinThickness'],histtype='stepfilled',bins=20)


# In[146]:


Positive['SkinThickness'].value_counts().head(7)


# In[147]:


plt.hist(Positive['Insulin'],histtype='stepfilled',bins=20)


# In[148]:


Positive['Insulin'].value_counts().head(7)


# In[149]:


#Scatter Plot


# In[150]:


BloodPressure = Positive['BloodPressure']
Glucose = Positive['Glucose']
SkinThickness = Positive['SkinThickness']
Insulin = Positive['Insulin']
BMI = Positive['BMI']


# In[151]:


plt.scatter(BloodPressure, Glucose, color=['b'])
plt.xlabel('BloodPressure')
plt.ylabel('Glucose')
plt.title('BloodPressure & Glucose')
plt.show()


# In[152]:


g =sns.scatterplot(x= "Glucose" ,y= "BloodPressure",
              hue="Outcome",
              data=data);


# In[153]:


B =sns.scatterplot(x= "BMI" ,y= "Insulin",
              hue="Outcome",
              data=data);


# In[154]:


S =sns.scatterplot(x= "SkinThickness" ,y= "Insulin",
              hue="Outcome",
              data=data);


# In[155]:


### correlation matrix
data.corr()


# In[156]:


### create correlation heat map
sns.heatmap(data.corr())


# In[157]:


plt.subplots(figsize=(8,8))
sns.heatmap(data.corr(),annot=True,cmap='viridis')  ### gives correlation value


# In[158]:


# Logistic Regreation and model building


# In[159]:


data.head(5)


# In[160]:


features = data.iloc[:,[0,1,2,3,4,5,6,7]].values
label = data.iloc[:,8].values


# In[161]:


#Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features,
                                                label,
                                                test_size=0.2,
                                                random_state =10)


# In[162]:


#Create model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train) 


# In[163]:


print(model.score(X_train,y_train))
print(model.score(X_test,y_test))


# In[164]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(label,model.predict(features))
cm


# In[165]:


from sklearn.metrics import classification_report
print(classification_report(label,model.predict(features)))


# In[166]:


#Preparing ROC Curve (Receiver Operating Characteristics Curve)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# predict probabilities
probs = model.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(label, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(label, probs)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')


# In[167]:


#Applying Decission Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model3 = DecisionTreeClassifier(max_depth=5)
model3.fit(X_train,y_train)


# In[168]:


model3.score(X_train,y_train)


# In[169]:


model3.score(X_test,y_test)


# In[170]:


#Applying Random Forest
from sklearn.ensemble import RandomForestClassifier
model4 = RandomForestClassifier(n_estimators=11)
model4.fit(X_train,y_train)


# In[171]:


model4.score(X_train,y_train)


# In[172]:


model4.score(X_test,y_test)


# In[177]:


#Support Vector Classifier

from sklearn.svm import SVC 
model5 = SVC(kernel='rbf',
           gamma='auto')
model5


# In[178]:


model5.fit(X_train,y_train)


# In[179]:


model5.score(X_test,y_test)


# In[181]:


model5.score(X_train,y_train)


# In[182]:


#Applying K-NN
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=7,
                             metric='minkowski',
                             p = 2)
model2.fit(X_train,y_train)


# In[183]:


#Preparing ROC Curve (Receiver Operating Characteristics Curve)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# predict probabilities
probs = model2.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(label, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(label, probs)
print("True Positive Rate - {}, False Positive Rate - {} Thresholds - {}".format(tpr,fpr,thresholds))
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")


# In[184]:


#Precision Recall Curve for Logistic Regression

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
# predict probabilities
probs = model.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model.predict(features)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(label, probs)
# calculate F1 score
f1 = f1_score(label, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(label, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall, precision, marker='.')


# In[185]:


#Precision Recall Curve for KNN

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
# predict probabilities
probs = model2.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model2.predict(features)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(label, probs)
# calculate F1 score
f1 = f1_score(label, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(label, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall, precision, marker='.')


# In[186]:


#Precision Recall Curve for Decission Tree Classifier

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
# predict probabilities
probs = model3.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model3.predict(features)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(label, probs)
# calculate F1 score
f1 = f1_score(label, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(label, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall, precision, marker='.')


# In[187]:


#Precision Recall Curve for Random Forest

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
# predict probabilities
probs = model4.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model4.predict(features)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(label, probs)
# calculate F1 score
f1 = f1_score(label, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(label, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall, precision, marker='.')


# In[ ]:




