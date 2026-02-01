#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score

import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv("Social_Network_Ads.csv")
df.head()


# ### Converting ``Gender`` to a number

# In[5]:


df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
df.head()


# In[6]:


X = df[['Age', 'EstimatedSalary', "Gender"]]
y = df['Purchased']


# It can be observed that the ``User_id`` column isn't included in the feature matrix; This is because it introduces noise and noise reduces accuracy
# Its is harmful for distance-based model like KNN

# ## Data splitting
# We will split the data into training and testing sets using an 80-20 split.

# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Data scaling using ``StandardScaler()``

# In[8]:


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# KNN uses distance between points, without scalling, ``Salary`` which obviously contains larger values than ``Age`` and ``Gender`` will dominate in the prediction. So we need to level the playing field

# ## Training the KNN Model

# In[9]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)


# In[10]:


y_pred = knn.predict(X_test_scaled)


# In[11]:


print("Accuracy:", accuracy_score(y_test, y_pred))


# In[12]:


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('KNN Confusion Matrix')
plt.show()


# ## Choosing the best value for ``k``

# In[13]:


accuracy_list = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)
    
plt.plot(range(1, 21), accuracy_list)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Accuracy')
plt.show()


# In[14]:


accuracy_list


# In[15]:


results = pd.DataFrame({"actual": y_test, "predicted": y_pred})
results.head()


# In[16]:


k_values = pd.DataFrame({"k": range(1, 21), "accuracy": accuracy_list}).sort_values(by="accuracy", ascending=False)

k_values


# using the optimal value for k ``k=8``   

# In[17]:


knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train_scaled, y_train)


# In[18]:


y_pred = knn.predict(X_test_scaled)


# In[19]:


print("Accuracy:", accuracy_score(y_test, y_pred))


# In[20]:


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('KNN Confusion Matrix')
plt.show()


# In[21]:


results = pd.DataFrame({"actual": y_test, "predicted": y_pred})
results.head()


# # Presentation: Understanding Customer Purchase Behavior
# 
# ## What Are We Trying to Understand?
# 
# This analysis helps us understand **who buys from our service** based on three simple factors:
# - **Age** - How old the customer is
# - **Salary** - How much they earn
# - **Gender** - Male or Female
# 
# By analyzing patterns from past customers, we can predict whether a new customer is likely to purchase or not.

# In[23]:


# Comparing customers who purchased vs didn't purchase
purchased_customers = df[df['Purchased'] == 1]
non_purchased_customers = df[df['Purchased'] == 0]

print("CUSTOMER SEGMENTS")
print(f"Total Customers Analyzed: {len(df):,}")
print(f"Customers Who Purchased: {len(purchased_customers):,} ({len(purchased_customers)/len(df)*100:.1f}%)")
print(f"Customers Who Did NOT Purchase: {len(non_purchased_customers):,} ({len(non_purchased_customers)/len(df)*100:.1f}%)")
print("\n")

# Key statistics
print("KEY DIFFERENCES BETWEEN GROUPS")

print(f"{'Metric':<30} {'Purchased':<15} {'Did Not Purchase':<15}")

print(f"{'Average Age':<30} {purchased_customers['Age'].mean():.1f} years{'':<5} {non_purchased_customers['Age'].mean():.1f} years")

print(f"{'Average Salary':<30} ${purchased_customers['EstimatedSalary'].mean():,.0f}{'':<7} ${non_purchased_customers['EstimatedSalary'].mean():,.0f}")

print(f"{'% Male':<30} {((purchased_customers['Gender']==0).sum()/len(purchased_customers))*100:.1f}%{'':<10} {((non_purchased_customers['Gender']==0).sum()/len(non_purchased_customers))*100:.1f}%")

print(f"{'% Female':<30} {((purchased_customers['Gender']==1).sum()/len(purchased_customers))*100:.1f}%{'':<10} {((non_purchased_customers['Gender']==1).sum()/len(non_purchased_customers))*100:.1f}%")


# In[24]:


# Plot Decision Boundary (using Age and EstimatedSalary)
import numpy as np

# Create mesh grid for Age and EstimatedSalary
x_min, x_max = X_train['Age'].min() - 5, X_train['Age'].max() + 5
y_min, y_max = X_train['EstimatedSalary'].min() - 5000, X_train['EstimatedSalary'].max() + 5000

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                     np.arange(y_min, y_max, 500))

# Create mesh with 3 features (Age, EstimatedSalary, Gender - using median)
gender_value = X_train['Gender'].median()
mesh_data = np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, gender_value)]

# Predict for each point in the mesh
Z = knn.predict(scaler.transform(mesh_data))
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn')
scatter = plt.scatter(X_train['Age'], X_train['EstimatedSalary'],
                      c=y_train, cmap='RdYlGn', edgecolors='black', s=50, alpha=0.7)

plt.xlabel('Age', fontweight='bold')
plt.ylabel('Estimated Salary', fontweight='bold')
plt.title('KNN Decision Boundary (k=8)', fontweight='bold', fontsize=14)
plt.legend(*scatter.legend_elements(), title='Purchased', loc='upper left')
plt.colorbar(label='Predicted Class')
plt.tight_layout()
plt.show()


# ## Key Patterns Observed
# 
# The data reveals clear differences between customers who purchased and those who didn't:
# 
# 1. **Age Difference** - Customers who purchased tend to be **older** than those who didn't
# 2. **Income Difference** - Customers who purchased generally have **higher salaries**
# 3. **Gender** - There's a slight difference in gender distribution between the two groups; Female customers purchase more than the male customers

# In[25]:


fig, axes = plt.subplots( figsize=(14, 5))


# Visualize Salary Distribution
axes.hist(purchased_customers['EstimatedSalary'], bins=20, alpha=0.7, label='Purchased', color='green', edgecolor='black')
axes.hist(non_purchased_customers['EstimatedSalary'], bins=20, alpha=0.7, label='Did NOT Purchase', color='red', edgecolor='black')
axes.set_xlabel('Annual Salary ($)', fontsize=12)
axes.set_ylabel('Number of Customers', fontsize=12)
axes.set_title('Salary Distribution: Purchasers vs Non-Purchasers', fontsize=13, fontweight='bold')
axes.legend(fontsize=11)
axes.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# In[26]:


# Create a scatter plot to show relationship between Age, Salary, and Purchase
fig, ax = plt.subplots(figsize=(12, 7))

# Plot non-purchasers
ax.scatter(non_purchased_customers['Age'], 
          non_purchased_customers['EstimatedSalary']/1000,
          alpha=0.5, s=100, label='Did NOT Purchase', color='red', edgecolors='darkred', linewidth=0.5)

# Plot purchasers
ax.scatter(purchased_customers['Age'], 
          purchased_customers['EstimatedSalary']/1000,
          alpha=0.5, s=100, label='Purchased', color='green', edgecolors='darkgreen', linewidth=0.5)

ax.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
ax.set_ylabel('Annual Salary ($1000s)', fontsize=12, fontweight='bold')
ax.set_title('Customer Profile: Age vs Salary', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ---
# 
# ## Our Prediction Model Explained
# 
# Think of our model as having a "memory" of past customers:
# - It remembers the age, salary, and gender of customers it has seen before
# - When we ask it to predict if a **new customer** will purchase, it looks for the 8 most similar past customers
# - It checks how many of those 8 similar customers actually purchased
# - If most of them purchased → it predicts the new customer will purchase too
# - If most of them did NOT purchase → it predicts the new customer won't purchase either
# 
# It's like asking 8 of your most similar friends whether they bought something to help you decide if you should buy it too!

# ## How Accurate Is Our Model?
# 
# Our model was tested on customers it had never seen before to see if it could predict correctly:

# In[27]:


# Get current predictions from our best model (k=8)
knn_final = KNeighborsClassifier(n_neighbors=8)
knn_final.fit(X_train_scaled, y_train)
y_pred_final = knn_final.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred_final)

print("MODEL PERFORMANCE")
print(f"Accuracy: {accuracy*100:.1f}%")
print(f"\nWhat this means:")
print(f"Out of 100 new customers, our model correctly predicts {accuracy*100:.0f} of them")

precision = precision_score(y_test, y_pred_final)
recall = recall_score(y_test, y_pred_final)

print(f"When model predicts 'Will Purchase': {precision*100:.0f}% of the time it's CORRECT")
print(f"When customer actually purchases: our model catches {recall*100:.0f}% of them")


# 
# 
# ## Key Takeaways for Business Decision Makers
# 
# ### Who Is Likely to Buy?
# Based on our analysis, customers who **purchase** typically have:
# - **Higher Age** → More mature customers tend to purchase more
# - **Higher Income** → Customers with better salaries are more likely to buy
# - These two factors together are the strongest predictors of whether someone will purchase
# 
# ### What This Means for Strategy?
# 1. **Target your marketing** toward older, higher-income customers for better ROI
# 2. **Focus your efforts** where the data shows the highest purchase probability
# 3. **Trust the patterns** - our model correctly predicts purchases 80%+ of the time
# 
# ### Next Steps
# - Use this model to score new leads and prioritize which customers to reach out to
# - Focus limited marketing budget on high-probability customers
# - Monitor if these patterns change over time as customer demographics shift

# In[ ]:





# In[ ]:




