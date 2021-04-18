#!/usr/bin/env python
# coding: utf-8

# ## 1) Choose a dataset

# In[1]:


# Import all the libraries required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


# In[2]:


# Import warnings and apply the setting to ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Read the dataset
df = pd.read_csv(r'C:\Users\vkarmarkar\OneDrive - University of Iowa\Desktop\Courses\ME4111 Scientific Computing and ML (Kamran)\Project\breast-cancer-wisconsin.csv',header=None)
df_names = pd.read_csv(r'C:\Users\vkarmarkar\OneDrive - University of Iowa\Desktop\Courses\ME4111 Scientific Computing and ML (Kamran)\Project\header_names.csv',header=None)
df.columns = df_names[0]


# ## 2) Explore the data set and select two or more features for data analysis

# ### a) Exploratory data analysis

# In[4]:


df.head() # Look at the first five rows


# In[5]:


df.tail() # Look at the last five rows


# In[6]:


df.shape # Look at the dimensions of the data set


# In[7]:


df.dtypes # Look at the data types of the different variables in the data set


# In[8]:


df.describe().T # Look at the summary statistics of the data set


# In[9]:


df.Class.unique() # Obtain the labels for benign and malignant tumours and check that there are just two labels


# In[10]:


df['Class'] = df['Class'].map({4:'M', 2:'B'}) # Temporarily transform the class variable to categorical variables


# In[11]:


df.Class.value_counts() # Obtain the total number of benign and malignant tumours after transformation


# In[12]:


sns.countplot(df['Class'], palette='husl') # Visualize number of benign and malignant tumours


# ### Remove columns which are not relevant or those which are not integers

# In[13]:


df.drop('Sample code number',axis=1, inplace=True)
df.drop('Bare Nuclei', axis=1, inplace=True)
df.head() # Look at first five rows after data cleaning


# In[14]:


df.tail() # Look at last five rows after data cleaning


# In[15]:


df.describe().T # Look at summary statistics after data cleaning


# ### b) Select two or more features for data analysis

# In[16]:


# Create a pairplot to visualize the variables that have good spatial separation between benign and malignant tumours
plt.figure(figsize=(12,6))
sns.pairplot(df, hue="Class")


# ### Perform further visualization on the variables that appear to have the best separation

# In[17]:


sns.violinplot(x='Class', y='Uniformity of Cell Size', data=df, palette='rocket')


# In[18]:


sns.violinplot(x='Class', y='Uniformity of Cell Shape', data=df, palette='rocket')


# In[19]:


sns.catplot(x='Class', y='Uniformity of Cell Size', kind='swarm', data=df)


# In[20]:


sns.catplot(x='Class', y='Uniformity of Cell Shape', kind='swarm', data=df)


# In[21]:


df['Class'] = df['Class'].map({'M':4, 'B':2}) # Convert the class variables back to integers


# In[22]:


df.head() # Visualize first five rows after the transformation


# In[23]:


df.dtypes # Check that all the data types are integers


# In[24]:


df.isnull().sum() # Check if the data set has any NaN values


# ### Split dataset based on the class of the tumour

# In[25]:


benign = df['Class'] == 2
malignant = df['Class'] == 4
df_benign = df[benign]
df_malignant = df[malignant]


# In[26]:


df_benign.describe().T # Look at the summary statistics for the benign tumour


# In[27]:


df_malignant.describe().T # Look at the summary statistics for the malignant tumour


# In[28]:


df_benign_mean = df_benign.mean() # Obtain the mean value of variables for benign tumour
df_benign_std = df_benign.std() # Obtain the standard deviation value of variables for benign tumour
df_malignant_mean = df_malignant.mean() # Obtain the mean value of variables for malignant tumour
df_malignant_std = df_malignant.std() # Obtain the standard deviation value of variables for malignant tumour


# ### Look at table of metrics that represent separation between the variables

# In[29]:


for k in range(len(df_malignant_mean)):
    print(k, df_malignant_mean[k]-df_benign_mean[k], df_benign_std[k], df_malignant_std[k])


# In[30]:


df.columns[2], df.columns[1] # Obtain the name of the variables that appear to have best separation


# ### Perform further visualization on the variables that appear to have the best separation

# In[31]:


sns.distplot(df_benign['Uniformity of Cell Shape'])


# In[32]:


sns.distplot(df_malignant['Uniformity of Cell Shape'], color='g')


# In[33]:


sns.distplot(df_benign['Uniformity of Cell Size'])


# In[34]:


sns.distplot(df_malignant['Uniformity of Cell Size'], color='g')


# In[35]:


fig, axs = plt.subplots(2)
dist = pd.DataFrame({'UCSHAPE_B':df_benign['Uniformity of Cell Shape'].values})
dist.plot.kde(ax=axs[0], legend=False, title='Histogram comparing first feature of Benign and Malignant Tumours')
dist.plot.hist(density=True, ax=axs[0])
dist_m = pd.DataFrame({'UCSHAPE_M':df_malignant['Uniformity of Cell Shape'].values})
dist_m.plot.kde(ax=axs[1], legend=False)
dist_m.plot.hist(density=True, ax=axs[1])
for j in range(2):
    axs[j].set_ylabel('Probability')
    axs[j].set_xlim(0,11)
    axs[j].grid(axis='y')
    axs[j].set_facecolor('#d8dcd6')


# In[36]:


fig, axs = plt.subplots(2)
dist = pd.DataFrame({'UCSIZE_B':df_benign['Uniformity of Cell Size'].values})
dist.plot.kde(ax=axs[0], legend=False, title='Histogram comparing second feature of Benign and Malignant Tumours')
dist.plot.hist(density=True, ax=axs[0])
dist_m = pd.DataFrame({'UCSIZE_M':df_malignant['Uniformity of Cell Size'].values})
dist_m.plot.kde(ax=axs[1], legend=False)
dist_m.plot.hist(density=True, ax=axs[1])
for j in range(2):
    axs[j].set_ylabel('Probability')
    axs[j].set_xlim(0,11)
    axs[j].grid(axis='y')
    axs[j].set_facecolor('#d8dcd6')


# In[37]:


plt.figure()
plt.scatter(df_benign['Uniformity of Cell Shape'], df_benign['Uniformity of Cell Size'], c='r', label='Benign')
plt.scatter(df_malignant['Uniformity of Cell Shape'], df_malignant['Uniformity of Cell Size'], c='b', label='Malignant')
plt.xlabel('feature Uniformity of Cell Shape (x1)', size=12)
plt.ylabel('feature Uniformity of Cell Size (x2)', size=12)
plt.title('Benign vs Malignant data', size= 12)
plt.legend()
plt.show()


# ## 3) Split the dataset into training, validation and test sets:

# In[53]:


X = df[['Uniformity of Cell Shape', 'Uniformity of Cell Size']].values
y_ = df['Class'].values
y = y_==2 # label value for benign tumour
X_t_, X_test_, y_t, y_test = train_test_split(X,y, random_state = 2) # training and test data split 25%
X_train_, X_val_, y_train, y_val = train_test_split(X,y, random_state = 2) # training and validation data split 25%
scaler = StandardScaler() # Perform scaling on variables to avoid ill conditioning


# In[39]:


# Tranform training, validation and test data based on standard scaler object
X_t = scaler.fit_transform(X_t_)
X_train = scaler.fit_transform(X_train_)
X_val = scaler.transform(X_val_)
X_test = scaler.fit_transform(X_test_)


# In[40]:


# Plot training data after scaling
x1_min, x1_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
x2_min, x2_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.scatter(X_train[y_train==1,0],X_train[y_train==1,1], marker = 'o', c='r', label='Benign')
plt.scatter(X_train[y_train==0,0],X_train[y_train==0,1], marker = 'o', c='b', label='Malignant')
plt.xlabel('feature Uniformity of Cell Shape (x1)', size=12)
plt.ylabel('feature Uniformity of Cell Size (x2)', size=12)
plt.title('Benign vs Malignant training data: After scaling', size= 12)
plt.legend()
plt.show()


# ## 4) and 5) Machine Learning and tuning of hyperparameters

# In[41]:


# Choosing hyperparameter C for LSVC using validation data set
C_val_list = [0.00001, 0.00005, 0.0001, 0.0002, 0.0005, 0.0007, 0.0008, 0.001, 0.01, 1.0, 2.0, 10.0, 100.0]
pre_list = []
rec_list = []
f1_list = []
train_acc_list = []
val_acc_list = []
for C_val in C_val_list:
    clf_LSVC = LinearSVC(C=C_val, max_iter=100000)
    clf_LSVC.fit(X_train, y_train)
    y_predict_LSVC = clf_LSVC.predict(X_val)
    pre_val = precision_score(y_true=y_val, y_pred=y_predict_LSVC)
    re_val = recall_score(y_true=y_val, y_pred=y_predict_LSVC)
    f1_val = f1_score(y_true=y_val, y_pred=y_predict_LSVC)
    train_acc_val = clf_LSVC.score(X_train, y_train)*100
    val_acc_val = clf_LSVC.score(X_val, y_val)*100
    train_acc_list.append(train_acc_val)
    val_acc_list.append(val_acc_val)
    rec_list.append(re_val)
    f1_list.append(f1_val)
    pre_list.append(pre_val)
df_metrics = pd.DataFrame(list(zip(C_val_list,train_acc_list, val_acc_list, rec_list, pre_list, f1_list)),
                  columns = ['C','training_acc', 'validation_acc', 'recall', 'precision', 'f1_score'])
df_metrics


# In[42]:


# Choosing hyperparameter gamma_RBF for C=1.0
C_val = 1.0 
gamma_rbf_list = [0.00001, 0.00005, 0.0001, 0.0002, 0.0005, 0.0006, 0.0007, 0.0008, 0.001, 0.005, 0.007, 0.01, 0.05, 0.1, 0.5, 1.0]
pre_list = []
rec_list = []
f1_list = []
train_acc_list = []
val_acc_list = []
for gamma_RBF_val in gamma_rbf_list:
    clf_RBF = SVC(kernel='rbf', random_state=0, gamma=gamma_RBF_val, C=C_val)
    clf_RBF.fit(X_train, y_train)
    y_predict_RBF = clf_RBF.predict(X_val)
    pre_val = precision_score(y_true=y_val, y_pred=y_predict_RBF)
    re_val = recall_score(y_true=y_val, y_pred=y_predict_RBF)
    f1_val = f1_score(y_true=y_val, y_pred=y_predict_RBF)
    train_acc_val = clf_RBF.score(X_train, y_train)*100
    val_acc_val = clf_RBF.score(X_val, y_val)*100
    train_acc_list.append(train_acc_val)
    val_acc_list.append(val_acc_val)
    rec_list.append(re_val)
    f1_list.append(f1_val)
    pre_list.append(pre_val)
df_metrics = pd.DataFrame(list(zip(gamma_rbf_list,train_acc_list, val_acc_list, rec_list, pre_list, f1_list)),
                  columns = ['gamma','training_acc', 'validation_acc', 'recall', 'precision', 'f1_score'])
df_metrics


# In[43]:


gamma_RBF = 0.0006 # Value selected based on hyperparameter tuning using validation data set (above)
C_val = 1.0 # Default value selected based on hyperparameter tuning using validation data set (above)
clf_LSVC = LinearSVC(C=C_val, max_iter=100000)
clf_LSVC.fit(X_train, y_train)
clf_RBF = SVC(kernel='rbf', random_state=0, gamma=gamma_RBF, C=C_val)
clf_RBF.fit(X_train, y_train)


# ### Plots for LSVC

# In[44]:


# Training data plot
x1_plot = np.arange(x1_min, x1_max, (x1_max-x1_min)/100.0)
x1_plot = x1_plot.reshape(-1, 1)
LSVC_plot = -(clf_LSVC.coef_[0,0]*x1_plot + clf_LSVC.intercept_[0])/clf_LSVC.coef_[0,1]
plt.plot(x1_plot, LSVC_plot, '-', c='black', label='LSVC decision boundary')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.scatter(X_train[y_train==1,0],X_train[y_train==1,1], marker = 'o', c='r', s=30, label='benign training')
plt.scatter(X_train[y_train==0,0],X_train[y_train==0,1], marker = 'o', c='b', s=30, label='malignant training')
plt.legend(scatterpoints=1, frameon=False, labelspacing=0.5)
plt.title('Bening/Malignant training data vs LSVC decision boundary', size=12)
plt.xlabel('feature Uniformity of Cell Shape (x1)', size=12)
plt.ylabel('feature Uniformity of Cell Size (x2)', size=12)
plt.legend(loc='upper left', scatterpoints=1, numpoints=1)
plt.show()


# In[45]:


# Test data plot
y_predict = clf_LSVC.predict(X_test)
plt.plot(x1_plot, LSVC_plot, '-', c='black', label='LSVC decision boundary')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.scatter(X_test[y_predict==1,0],X_test[y_predict==1,1], marker = '+', c='r', s=200, label='Benign prediction')
plt.scatter(X_test[y_predict==0,0],X_test[y_predict==0,1], marker = '+', c='b', s=200, label='Malignant prediction')
plt.scatter(X_test[y_test==1,0],X_test[y_test==1,1], marker = 'o', c='r', s=30, label='Benign test')
plt.scatter(X_test[y_test==0,0],X_test[y_test==0,1], marker = 'o', c='b', s=30, label='Malignant test')
plt.legend(scatterpoints=1, frameon=False, labelspacing=0.5)
plt.title('Benign/Malignant test data vs LSVC', size=12)
plt.xlabel('feature Uniformity of Cell Shape (x1)', size=12)
plt.ylabel('feature Uniformity of Cell Size (x2)', size=12)
plt.legend(loc='upper left', scatterpoints=1, numpoints=1)
plt.show()


# ### SVC with RBF plots

# In[46]:


# Training data plots
plt.figure()
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
h = .02
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
Z = clf_RBF.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired, shading='auto')
plt.scatter(X_train[y_train==1,0],X_train[y_train==1,1], marker = 'o', c='r', label='Benign training')
plt.scatter(X_train[y_train==0,0],X_train[y_train==0,1], marker = 'o', c='b', label='Malignant training')
plt.title('Benign vs Malignant training data and decision regions')
plt.xlabel('feature Uniformity of Cell Shape (x1)')
plt.ylabel('feature Uniformity of Cell Size (x2)')
plt.legend(loc='upper left', scatterpoints=1, numpoints=1)
plt.show()


# In[47]:


# Test data plots
y_pred = clf_RBF.predict(X_test)
plt.figure()
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired, shading='auto')
plt.scatter(X_test[y_test==1,0],X_test[y_test==1,1], marker = 'o', c='r', label='Benign test')
plt.scatter(X_test[y_test==0,0],X_test[y_test==0,1], marker = 'o', c='b', label='Malignant test')
plt.scatter(X_test[y_pred==1,0],X_test[y_pred==1,1], marker = '+', c='r', s=200, label='Benign pred')
plt.scatter(X_test[y_pred==0,0],X_test[y_pred==0,1], marker = '+', c='b', s=200, label='Malignant pred')
plt.title('Benign vs Malignant decision regions Gamma: {:.4f}; \n SVC train score: {:.3f}% and test score {:.3f}%'
          .format(gamma_RBF, clf_RBF.score(X_train, y_train)*100, clf_RBF.score(X_test, y_test)*100))
plt.xlabel('feature Uniformity of Cell Shape (x1)')
plt.ylabel('feature Uniformity of Cell Size (x2)')
plt.legend(loc='upper left', scatterpoints=1, numpoints=1)
plt.show()


# ## 6) Use the test set to conduct the final evaluation of the selected best models, and choose a better one as the final decision

# In[48]:


len(X_test) # Compute total number of members in the test data set


# ### LSVC report

# In[49]:


y_predict_LSVC = clf_LSVC.predict(X_test)
confmat_LSVC = confusion_matrix(y_true=y_test, y_pred=y_predict_LSVC)
print("Confusion Matrix for LSVC below")
print(confmat_LSVC)
print('Precision for LSVC: %.3f' % precision_score(y_true=y_test, y_pred=y_predict_LSVC))
print('Recall for LSVC: %.3f' % recall_score(y_true=y_test, y_pred=y_predict_LSVC))
print('F1 score for LSVC: %.3f' % f1_score(y_true=y_test, y_pred=y_predict_LSVC))
print('Accuracy of LSVC classifier on training set: {:.2f}%'
     .format(clf_LSVC.score(X_train, y_train)*100))
print('Accuracy of LSVC classifier on validation set: {:.2f}%'
     .format(clf_LSVC.score(X_val, y_val)*100))
print('Accuracy of LSVC classifier on test set: {:.2f}%'
     .format(clf_LSVC.score(X_test, y_test)*100))


# In[50]:


sns.heatmap(confmat_LSVC, annot=True) # Heatmap to visualize confusion matrix for LSVC


# ### SVC with RBF kernel report

# In[51]:


y_predict_RBF = clf_RBF.predict(X_test)
confmat_RBF = confusion_matrix(y_true=y_test, y_pred=y_predict_RBF)
print("Confusion Matrix for SVC with RBF kernel below")
print(confmat_RBF)
print('Precision for SVC with RBF kernel: %.3f' % precision_score(y_true=y_test, y_pred=y_predict_RBF))
print('Recall for SVC with RBF kernel: %.3f' % recall_score(y_true=y_test, y_pred=y_predict_RBF))
print('F1 score for SVC with RBF kernel: %.3f' % f1_score(y_true=y_test, y_pred=y_predict_RBF))
print('Accuracy of SVF with RBF kernel classifier on training set: {:.2f}%'
     .format(clf_RBF.score(X_train, y_train)*100))
print('Accuracy of SVC with RBF classifier on validation set: {:.2f}%'
     .format(clf_RBF.score(X_val, y_val)*100))
print('Accuracy of SVC with RBF kernel classifier on test set: {:.2f}%'
     .format(clf_RBF.score(X_test, y_test)*100))


# In[52]:


sns.heatmap(confmat_RBF, annot=True) # Heatmap to visualize confusion matrix for SVC with RBF kernel

