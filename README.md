### EX4 Implementation of Cluster and Visitor Segmentation for Navigation patterns
### DATE: 07-02-2026
### AIM: To implement Cluster and Visitor Segmentation for Navigation patterns in Python.
### Description:
<div align= "justify">Cluster visitor segmentation refers to the process of grouping or categorizing visitors to a website, 
  application, or physical location into distinct clusters or segments based on various characteristics or behaviors they exhibit. 
  This segmentation allows businesses or organizations to better understand their audience and tailor their strategies, marketing efforts, 
  or services to meet the specific needs and preferences of each cluster.</div>
  
### Procedure:
1) Read the CSV file: Use pd.read_csv to load the CSV file into a pandas DataFrame.
2) Define Age Groups by creating a dictionary containing age group conditions using Boolean conditions.
3) Segment Visitors by iterating through the dictionary and filter the visitors into respective age groups.
4) Visualize the result using matplotlib.

### Program:
```python
import pandas as pd
visitor_df = pd.read_csv('/content/clustervisitor.csv')

age_groups = {
    'Young': visitor_df['Age'] <= 30,
    'Middle-aged': (visitor_df['Age'] > 30) & (visitor_df['Age'] <= 50),
    'Elderly': visitor_df['Age'] > 50
}

for group, condition in age_groups.items():  
    visitors_in_group = visitor_df[condition]
    printf("count= {len(visitors_in_group)}")
    print(f"Visitors in {group} age group:")
    print(visitors_in_group)



```
### Output:

<img width="497" height="813" alt="image" src="https://github.com/user-attachments/assets/b647dedc-07d7-4be0-ae31-1dee22d7211c" />



### Visualization:
```python
import matplotlib.pyplot as plt
# Create a list to store counts of visitors in each age group
visitor_counts=[]

for group,condition in age_groups.items():
  visitors_in_group=visitor_df[condition]
  visitor_counts.append(len(visitors_in_group)


age_group_labels=list(age_groups.keys())
plt.figure(figsize=(8, 6))
plt.bar(age_group_labels, visitor_counts, color='skyblue')
plt.xlabel('Age Groups')
plt.ylabel('Number of Visitors')
plt.title('Visitor Distribution Across Age Groups')
plt.show()

```
### Output:

<img width="878" height="686" alt="image" src="https://github.com/user-attachments/assets/a1d099db-acca-449a-8891-6fc80728d9b4" />

### Program 2:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Clean column names (Excel safety)
df.columns = df.columns.str.strip()

# Remove existing Cluster column if present
if 'Cluster' in df.columns:
    df = df.drop(columns=['Cluster'])

# Select features for clustering
X = df[['Age', 'Income']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Convert cluster labels to DataFrame
cluster_df = pd.DataFrame(clusters, columns=['Cluster'])

# Concatenate cluster labels with original dataset
df = pd.concat([df, cluster_df], axis=1)

# -----------------------------------
# Arrange clusters by average Income
# -----------------------------------
cluster_order = (
    df.groupby('Cluster')['Income']
    .mean()
    .sort_values()
    .index
)

cluster_mapping = {old: new for new, old in enumerate(cluster_order)}
df['Cluster'] = df['Cluster'].map(cluster_mapping)

# -----------------------------------
# Display clustering result neatly
# -----------------------------------
print("\nK-Means Clustering Result (Ordered by Income)")
print(df[['Age', 'Income', 'Cluster']].sort_values('Cluster').to_string(index=False))

# -----------------------------------
# Visualize clusters
# -----------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(df['Age'], df['Income'], c=df['Cluster'])
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("K-Means Clustering using Age and Income (Low → High)")
plt.show()
```

### Output:

<img width="507" height="709" alt="image" src="https://github.com/user-attachments/assets/f107f451-d0b4-4d35-a9d6-19a322b3dfde" />


<img width="866" height="688" alt="image" src="https://github.com/user-attachments/assets/38a04848-8c66-4841-b502-e8772550f858" />


### Result:

Thus the cluster and visitor segmentation for navigation patterns was implemented successfully in python.
