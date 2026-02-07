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


### Result:

Thus the cluster and visitor segmentation for navigation patterns was implemented successfully in python.
