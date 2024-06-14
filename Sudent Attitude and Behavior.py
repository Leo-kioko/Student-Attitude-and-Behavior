#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# In[3]:


pd.read_csv(r"C:\Users\Noel\Downloads\Pandas\Student Attitude and Behavior.csv")


# In[4]:


pd.set_option('display.max_rows',None)


# In[8]:


df= pd.read_csv(r"C:\Users\Noel\Downloads\Pandas\Student Attitude and Behavior.csv")
df


# In[9]:


# Ensure the relevant columns exist in the dataframe
required_columns = ['Certification Course', '10th Mark', '12th Mark', 'college mark']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Define a function to create a bar plot for academic performance based on certification completion
def plot_academic_performance(df, mark_column, ax):
    sns.boxplot(data=df, x='Certification Course', y=mark_column, ax=ax)
    ax.set_title(f'Effect of Certification on {mark_column}')
    ax.set_xlabel('Certification Course')
    ax.set_ylabel(mark_column)

# Set up the plot
fig, axes = plt.subplots(3, 1, figsize=(12, 18))

# Plot academic performance for 10th Mark, 12th Mark, and College Mark
plot_academic_performance(df, '10th Mark', axes[0])
plot_academic_performance(df, '12th Mark', axes[1])
plot_academic_performance(df, 'college mark', axes[2])

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()


# In[10]:


# Ensure the relevant columns exist in the dataframe
required_columns = ['Gender', '10th Mark', '12th Mark', 'college mark']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Define a function to create a box plot for academic performance based on gender
def plot_academic_performance(df, mark_column, ax):
    sns.boxplot(data=df, x='Gender', y=mark_column, ax=ax)
    ax.set_title(f'Academic Performance in {mark_column} by Gender')
    ax.set_xlabel('Gender')
    ax.set_ylabel(mark_column)

# Set up the plot
fig, axes = plt.subplots(3, 1, figsize=(12, 18))

# Plot academic performance for 10th Mark, 12th Mark, and College Mark
plot_academic_performance(df, '10th Mark', axes[0])
plot_academic_performance(df, '12th Mark', axes[1])
plot_academic_performance(df, 'college mark', axes[2])

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()


# In[11]:


# Ensure the relevant columns exist in the dataframe
required_columns = ['Gender', '10th Mark', '12th Mark', 'college mark']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Compute the average marks for male and female students
average_marks = df.groupby('Gender')[['10th Mark', '12th Mark', 'college mark']].mean().reset_index()

# Set up the plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Define a function to create a bar plot for academic performance based on gender
def plot_average_marks(df, mark_column, ax):
    sns.barplot(data=df, x='Gender', y=mark_column, ax=ax, palette='viridis')
    ax.set_title(f'Average {mark_column} by Gender')
    ax.set_xlabel('Gender')
    ax.set_ylabel(mark_column)

# Plot average marks for 10th Mark, 12th Mark, and College Mark
plot_average_marks(average_marks, '10th Mark', axes[0])
plot_average_marks(average_marks, '12th Mark', axes[1])
plot_average_marks(average_marks, 'college mark', axes[2])

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()


# In[18]:



# Ensure the relevant columns exist in the dataframe
required_columns = ['Department', 'hobbies', '10th Mark', '12th Mark', 'college mark']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Melt the dataframe to have a long-form data suitable for seaborn
melted_df = pd.melt(df, id_vars=['Department', 'hobbies'], value_vars=['10th Mark', '12th Mark', 'college mark'],
                    var_name='Academic Stage', value_name='Mark')

# Set up the plot
plt.figure(figsize=(18, 10))

# Create a bar plot showing average marks by department and hobbies
sns.barplot(data=melted_df, x='Department', y='Mark', hue='hobbies', ci=None, palette='viridis', dodge=True)

# Set labels and title
plt.title('Comparison of Academic Performance by Department and Hobbies')
plt.xlabel('Department')
plt.ylabel('Average Mark')
plt.legend(title='Hobbies', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot
plt.tight_layout()
plt.show()


# In[20]:


# Ensure the relevant columns exist in the dataframe
required_columns = ['Gender', 'Height(CM)', 'Weight(KG)']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Set up the plot
plt.figure(figsize=(12, 6))

# Create a scatter plot with regression lines
sns.lmplot(data=df, x='Height(CM)', y='Weight(KG)', hue='Gender', aspect=1.5, height=6, markers=['o', 's'])

# Set labels and title
plt.title('Correlation between Height and Weight by Gender')
plt.xlabel('Height(CM)')
plt.ylabel('Weight(KG)')

# Show plot
plt.tight_layout()
plt.show()


# In[28]:


# Correct the column name
df.rename(columns={'daily studing time': 'Daily Studying Time'}, inplace=True)

# Ensure the relevant columns exist in the dataframe
required_columns = ['Height(CM)', 'Weight(KG)', '10th Mark', '12th Mark', 'college mark', 'Daily Studying Time']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Set up the plot
plt.figure(figsize=(12, 8))

# Create a scatter plot
sns.scatterplot(data=df, x='Height(CM)', y='Weight(KG)', hue='Daily Studying Time', size='Daily Studying Time',
                palette='viridis', sizes=(50, 200), alpha=0.7)

# Set labels and title
plt.title('Correlation between Height, Weight, Academic Performance, and Daily Study Time')
plt.xlabel('Height(CM)')
plt.ylabel('Weight(KG)')
plt.legend(title='Daily Studying Time')

# Show plot
plt.tight_layout()
plt.show()


# In[31]:


# Ensure the relevant columns exist in the dataframe
required_columns = ['hobbies', '10th Mark', '12th Mark', 'college mark', 'part-time job']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Set up the plot
plt.figure(figsize=(18, 6))

# Create a scatter plot for hobbies and academic performance
sns.scatterplot(data=df, x='hobbies', y='10th Mark', hue='part-time job', palette='viridis', alpha=0.7)
sns.scatterplot(data=df, x='hobbies', y='12th Mark', hue='part-time job', palette='viridis', alpha=0.7)
sns.scatterplot(data=df, x='hobbies', y='college mark', hue='part-time job', palette='viridis', alpha=0.7)

# Set labels and title
plt.title('Relationship between Hobbies, Academic Performance, and Part-Time Job')
plt.xlabel('Hobbies')
plt.ylabel('Academic Performance')
plt.legend(title='Part-Time Job')

# Show plot
plt.tight_layout()
plt.show()


# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# In[10]:


df= pd.read_csv(r"C:\Users\Noel\Downloads\Pandas\Student Attitude and Behavior.csv")
df


# In[11]:


# Ensure the relevant column exists in the dataframe
if 'Daily Study Time' not in df.columns:
    raise ValueError("Column 'Daily Study Time' is missing from the dataset")

# Define a mapping from the descriptive ranges to numeric values
study_time_mapping = {
    '0 - 30 minute': 0.25,
    '30 - 60 minute': 0.75,
    '1 - 2 Hour': 1.5,
    '2 - 3 hour': 2.5,
    '3 - 4 hour': 3.5,
    'More Than 4 hour': 4.5
}

# Convert the Daily Study Time column to numeric values using the mapping
df['Daily Study Time'] = df['Daily Study Time'].map(study_time_mapping)

# Calculate the average daily study time
average_study_time = df['Daily Study Time'].mean()

# Print the average study time
print(f'The average daily study time is {average_study_time:.2f} hours')


# In[19]:


# Ensure the relevant columns exist in the dataframe
required_columns = ['Stress Levels', '10th Mark', '12th Mark', 'College Mark']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' is missing from the dataset")

# Convert columns to numeric
for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Plotting the pairplot
sns.pairplot(df[required_columns], diag_kind='kde')
plt.suptitle('Pairplot of Stress Levels and Academic Performance', y=1.02)
plt.show()

# Compute and print correlation coefficients
correlation_matrix = df[required_columns].corr()
print("Correlation Matrix:")
print(correlation_matrix)


# In[27]:


# Ensure the relevant columns exist in the dataframe
required_columns = ['Stress Levels', '10th Mark', '12th Mark', 'College Mark']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Define mapping for Stress Levels categories to numerical values
stress_level_mapping = {
    'Awful': 0,
    'Bad': 1,
    'Good': 2,
    'Fabulous': 3
}

# Map Stress Levels to numerical values
df['Stress Levels'] = df['Stress Levels'].map(stress_level_mapping)

# Set up the plot
plt.figure(figsize=(10, 8))

# Create a heatmap to visualize correlation
sns.heatmap(df[['Stress Levels', '10th Mark', '12th Mark', 'College Mark']].corr(), annot=True, cmap='coolwarm', linewidths=0.5)

# Set title and display plot
plt.title('Correlation between Stress Levels and Academic Performance')
plt.show()


# In[37]:


# Ensure the relevant columns exist in the dataframe
required_columns = ['Stress Levels', '10th Mark', '12th Mark', 'College Mark']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Convert 'Stress Levels' column to object type to fill missing values
df['Stress Levels'] = df['Stress Levels'].astype(object)

# Replace missing values with 0
df.fillna(0, inplace=True)

# Convert 'Stress Levels' back to categorical type and map categories to numerical values
stress_level_mapping = {'Awful': 0, 'Bad': 1, 'Good': 2, 'Fabulous': 3}
df['Stress Levels'] = df['Stress Levels'].map(stress_level_mapping)

# Set up the plot
plt.figure(figsize=(12, 6))

# Create bar plots for each academic performance metric
sns.barplot(data=df, x='Stress Levels', y='10th Mark', label='10th Mark', ci=None, color='blue', alpha=0.6)
sns.barplot(data=df, x='Stress Levels', y='12th Mark', label='12th Mark', ci=None, color='green', alpha=0.6)
sns.barplot(data=df, x='Stress Levels', y='College Mark', label='College Mark', ci=None, color='red', alpha=0.6)

# Set labels and title
plt.title('Correlation between Stress Levels and Academic Performance')
plt.xlabel('Stress Levels')
plt.ylabel('Academic Performance (%)')
plt.xticks([0, 1, 2, 3], ['Awful', 'Bad', 'Good', 'Fabulous'])  # Replace numerical values with categorical labels
plt.ylim(0, 100)  # Set y-axis limits to reflect the percentage range
plt.legend()

# Show plot
plt.tight_layout()
plt.show()


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# In[2]:


df = pd.read_csv(r"C:\Users\Noel\Downloads\Pandas\Student Attitude and Behavior.csv")
df


# In[5]:


# Ensure the relevant columns exist in the dataframe
required_columns = ['Stress Levels', '10th Mark', '12th Mark', 'College Mark']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Replace missing values with 0
df.fillna(0, inplace=True)

# Map categories to numerical values
stress_level_mapping = {'Awful': 0, 'Bad': 1, 'Good': 2, 'Fabulous': 3}
df['Stress Levels'] = df['Stress Levels'].map(stress_level_mapping)

# Ensure the mapping was successful
if df['Stress Levels'].isnull().any():
    raise ValueError("There are values in 'Stress Levels' that were not mapped successfully.")

# Set up the plot
plt.figure(figsize=(12, 6))

# Create bar plots for each academic performance metric
sns.barplot(data=df, x='Stress Levels', y='10th Mark', label='10th Mark', ci=None, color='blue', alpha=0.6)
sns.barplot(data=df, x='Stress Levels', y='12th Mark', label='12th Mark', ci=None, color='green', alpha=0.6)
sns.barplot(data=df, x='Stress Levels', y='College Mark', label='College Mark', ci=None, color='red', alpha=0.6)

# Set labels and title
plt.title('Correlation between Stress Levels and Academic Performance')
plt.xlabel('Stress Levels')
plt.ylabel('Academic Performance (%)')
plt.xticks([0, 1, 2, 3], ['Awful', 'Bad', 'Good', 'Fabulous'])  # Replace numerical values with categorical labels
plt.ylim(0, 100)  # Set y-axis limits to reflect the percentage range
plt.legend()

# Show plot
plt.tight_layout()
plt.show()


# In[13]:


# Ensure the relevant columns exist in the dataframe
required_columns = ['Stress Levels', '10th Mark', '12th Mark', 'College Mark']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Replace missing values with 0
df.fillna(0, inplace=True)

# Map categories to numerical values
stress_level_mapping = {'Awful': 0, 'Bad': 1, 'Good': 2, 'fabulous': 3}
df['Stress Levels'] = df['Stress Levels'].map(stress_level_mapping)

# Check for unmapped values in 'Stress Levels'
unmapped_values = df['Stress Levels'][df['Stress Levels'].isnull()].unique()
if len(unmapped_values) > 0:
    print(f"The following values in 'Stress Levels' were not mapped successfully: {unmapped_values}")
    raise ValueError("There are values in 'Stress Levels' that were not mapped successfully.")

# Set up the plot
plt.figure(figsize=(12, 6))

# Create bar plots for each academic performance metric
sns.barplot(data=df, x='Stress Levels', y='10th Mark', ci=None, color='blue', alpha=0.6, label='10th Mark')
sns.barplot(data=df, x='Stress Levels', y='12th Mark', ci=None, color='green', alpha=0.6, label='12th Mark')
sns.barplot(data=df, x='Stress Levels', y='College Mark', ci=None, color='red', alpha=0.6, label='College Mark')

# Set labels and title
plt.title('Correlation between Stress Levels and Academic Performance')
plt.xlabel('Stress Levels')
plt.ylabel('Academic Performance (%)')
plt.xticks([0, 1, 2, 3], ['Awful', 'Bad', 'Good', 'fabulous'])  # Replace numerical values with categorical labels
plt.ylim(0, 100)  # Set y-axis limits to reflect the percentage range
plt.legend()

# Show plot
plt.tight_layout()
plt.show()


# In[9]:


# Ensure the relevant columns exist in the dataframe
required_columns = ['Indoor studying', '10th Mark', '12th Mark', 'College Mark']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Replace missing values with 0 in academic performance columns
df[['10th Mark', '12th Mark', 'College Mark']] = df[['10th Mark', '12th Mark', 'College Mark']].fillna(0)

# Check the unique values in 'Indoor Studying' column
print("Unique values in 'Indoor studying' column:", df['Indoor studying'].unique())

# Convert 'Indoor Studying' to categorical if it's not already
if not pd.api.types.is_categorical_dtype(df['Indoor studying']):
    df['Indoor studying'] = df['Indoor studying'].astype('category')

# Set up the plot
plt.figure(figsize=(12, 6))

# Create bar plots for each academic performance metric
sns.barplot(data=df, x='Indoor studying', y='10th Mark', ci=None, color='blue', alpha=0.6, label='10th Mark')
sns.barplot(data=df, x='Indoor studying', y='12th Mark', ci=None, color='green', alpha=0.6, label='12th Mark')
sns.barplot(data=df, x='Indoor studying', y='College Mark', ci=None, color='red', alpha=0.6, label='College Mark')

# Set labels and title
plt.title('Relationship between Indoor studying and Academic Performance')
plt.xlabel('Indoor studying')
plt.ylabel('Academic Performance (%)')
plt.ylim(0, 100)  # Set y-axis limits to reflect the percentage range
plt.legend()

# Show plot
plt.tight_layout()
plt.show()


# In[10]:


# Sample data
data = {
    'Daily Study Time': ['0 - 30 minute', '30 - 60 minute', '1 - 2 Hour', '2 - 3 hour', '3 - 4 hour', 'More Than 4 hour'],
    '10th Mark': [85, 88, 90, 75, 95, 80],
    '12th Mark': [82, 87, 92, 78, 94, 85],
    'College Mark': [80, 85, 91, 76, 93, 82]
}

df = pd.DataFrame(data)

# Map study time categories to numerical values
study_time_mapping = {
    '0 - 30 minute': 0,
    '30 - 60 minute': 1,
    '1 - 2 Hour': 2,
    '2 - 3 hour': 3,
    '3 - 4 hour': 4,
    'More Than 4 hour': 5
}

df['Daily Study Time Num'] = df['Daily Study Time'].map(study_time_mapping)

# Ensure the mapping was successful
if df['Daily Study Time Num'].isnull().any():
    unmapped_values = df['Daily Study Time'][df['Daily Study Time Num'].isnull()].unique()
    print(f"The following values in 'Daily Study Time' were not mapped successfully: {unmapped_values}")
    raise ValueError("There are values in 'Daily Study Time' that were not mapped successfully.")

# Set up the plot
plt.figure(figsize=(14, 8))

# Create bar plots for each academic performance metric
sns.barplot(data=df, x='Daily Study Time Num', y='10th Mark', ci=None, color='blue', alpha=0.6, label='10th Mark')
sns.barplot(data=df, x='Daily Study Time Num', y='12th Mark', ci=None, color='green', alpha=0.6, label='12th Mark')
sns.barplot(data=df, x='Daily Study Time Num', y='College Mark', ci=None, color='red', alpha=0.6, label='College Mark')

# Set labels and title
plt.title('Relationship between Daily Study Time and Academic Performance')
plt.xlabel('Daily Study Time')
plt.ylabel('Academic Performance (%)')
plt.xticks([0, 1, 2, 3, 4, 5], ['0-30 min', '30-60 min', '1-2 hrs', '2-3 hrs', '3-4 hrs', '4+ hrs'])
plt.ylim(0, 100)  # Set y-axis limits to reflect the percentage range
plt.legend()

# Show plot
plt.tight_layout()
plt.show()


# In[11]:


# Ensure the relevant columns exist in the dataframe
required_columns = ['Daily Study Time', '10th Mark', '12th Mark', 'College Mark']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Map Daily Study Time categories to numeric values
study_time_mapping = {
    '0 - 30 minute': 0.25,
    '30 - 60 minute': 0.75,
    '1 - 2 Hour': 1.5,
    '2 - 3 hour': 2.5,
    '3 - 4 hour': 3.5,
    'More Than 4 hour': 4.5
}
df['Daily Study Time'] = df['Daily Study Time'].map(study_time_mapping)

# Check for unmapped values in 'Daily Study Time'
unmapped_values = df['Daily Study Time'][df['Daily Study Time'].isnull()].unique()
if len(unmapped_values) > 0:
    print(f"The following values in 'Daily Study Time' were not mapped successfully: {unmapped_values}")
    raise ValueError("There are values in 'Daily Study Time' that were not mapped successfully.")

# Set up the plot
plt.figure(figsize=(14, 10))

# Create scatter plots for each academic performance metric
plt.subplot(3, 1, 1)
sns.scatterplot(data=df, x='Daily Study Time', y='10th Mark')
plt.title('Daily Study Time vs 10th Mark')
plt.xlabel('Daily Study Time (hours)')
plt.ylabel('10th Mark (%)')
plt.ylim(0, 100)

plt.subplot(3, 1, 2)
sns.scatterplot(data=df, x='Daily Study Time', y='12th Mark')
plt.title('Daily Study Time vs 12th Mark')
plt.xlabel('Daily Study Time (hours)')
plt.ylabel('12th Mark (%)')
plt.ylim(0, 100)

plt.subplot(3, 1, 3)
sns.scatterplot(data=df, x='Daily Study Time', y='College Mark')
plt.title('Daily Study Time vs College Mark')
plt.xlabel('Daily Study Time (hours)')
plt.ylabel('College Mark (%)')
plt.ylim(0, 100)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()


# In[12]:


# Ensure the relevant columns exist in the dataframe
required_columns = ['Stress Levels', '10th Mark', '12th Mark', 'College Mark']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Replace missing values in Stress Levels with 'Awful'
df['Stress Levels'].fillna('Awful', inplace=True)

# Map Stress Levels to numeric values
stress_level_mapping = {
    'Awful': 0,
    'Bad': 1,
    'Good': 2,
    'fabulous': 3
}
df['Stress Levels'] = df['Stress Levels'].map(stress_level_mapping)

# Check for unmapped values in 'Stress Levels'
unmapped_values = df['Stress Levels'][df['Stress Levels'].isnull()].unique()
if len(unmapped_values) > 0:
    print(f"The following values in 'Stress Levels' were not mapped successfully: {unmapped_values}")
    raise ValueError("There are values in 'Stress Levels' that were not mapped successfully.")

# Set up the plot
plt.figure(figsize=(14, 10))

# Create scatter plots for each academic performance metric
plt.subplot(3, 1, 1)
sns.scatterplot(data=df, x='Stress Levels', y='10th Mark')
plt.title('Stress Levels vs 10th Mark')
plt.xlabel('Stress Levels')
plt.ylabel('10th Mark (%)')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Awful', 'Bad', 'Good', 'fabulous'])
plt.ylim(0, 100)

plt.subplot(3, 1, 2)
sns.scatterplot(data=df, x='Stress Levels', y='12th Mark')
plt.title('Stress Levels vs 12th Mark')
plt.xlabel('Stress Levels')
plt.ylabel('12th Mark (%)')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Awful', 'Bad', 'Good', 'fabulous'])
plt.ylim(0, 100)

plt.subplot(3, 1, 3)
sns.scatterplot(data=df, x='Stress Levels', y='College Mark')
plt.title('Stress Levels vs College Mark')
plt.xlabel('Stress Levels')
plt.ylabel('College Mark (%)')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Awful', 'Bad', 'Good', 'fabulous'])
plt.ylim(0, 100)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()


# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
pd.set_option('display.max_rows',None)


# In[16]:


df = pd.read_csv(r"C:\Users\Noel\Downloads\Pandas\Student Attitude and Behavior.csv")
df


# In[17]:


# Ensure the 'Degree preference' column exists in the dataframe
if 'Degree preference' not in df.columns:
    raise ValueError("Column 'Degree preference' is missing from the dataset")

# Count the occurrences of each category in the 'Degree preference' column
degree_preference_counts = df['Degree preference'].value_counts()

# Calculate the percentage of students who like their degree (answered 'Yes')
percentage_liked_degree = (degree_preference_counts['Yes'] / df.shape[0]) * 100

print(f"The percentage of students who like their degree is: {percentage_liked_degree:.2f}%")


# In[21]:


# Ensure the required columns exist in the dataframe
required_columns = ['Willingness', 'Degree preference']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Convert the 'Willingness' column from percent form to numeric
df['Willingness'] = df['Willingness'].str.rstrip('%').astype(float)

# Group the data by 'Degree preference' and calculate the average 'Willingness' for each group
grouped_data = df.groupby('Degree preference')['Willingness'].mean()

# Plotting
plt.figure(figsize=(8, 6))
grouped_data.plot(kind='bar', color='skyblue', alpha=0.7)

# Set plot labels and title
plt.title('Willingness to Pursue a Career by Degree Preference')
plt.xlabel('Degree Preference')
plt.ylabel('Average Willingness (%)')

# Set x-axis tick labels
plt.xticks(range(len(grouped_data)), grouped_data.index)

# Show plot
plt.tight_layout()
plt.show()


# In[22]:


# Ensure the required columns exist in the dataframe
required_columns = ['Department', 'Gender', 'College Mark', 'Salary Expectation']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Group the data by 'Department', 'Gender', and 'College Mark', and calculate the average 'Salary Expectations'
grouped_data = df.groupby(['Department', 'Gender', 'College Mark'])['Salary Expectation'].mean().reset_index()

# Plotting using seaborn
plt.figure(figsize=(12, 8))
sns.barplot(data=grouped_data, x='Department', y='Salary Expectation', hue='Gender', ci=None)

# Set plot labels and title
plt.title('Salary Expectations by Department, Gender, and College Mark')
plt.xlabel('Department')
plt.ylabel('Average Salary Expectations')

# Show plot
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[30]:


# Ensure the required columns exist in the dataframe
required_columns = ['Salary Expectation', 'Department', 'Gender', '10th Mark']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Plotting the heatmap
plt.figure(figsize=(12, 8))
heatmap_data = df.pivot_table(index='Department', columns=['Gender', '10th Mark'], values='Salary Expectation', aggfunc='mean')
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=".1f", linewidths=.5)

# Set plot labels and title
plt.title('Salary Expectations by Department, Gender, and 10th Mark')
plt.xlabel('Gender and 10th Mark')
plt.ylabel('Department')

# Show plot
plt.tight_layout()
plt.show()


# In[35]:


# Ensure the required columns exist in the dataframe
required_columns = ['Department', 'Gender', '12th Mark', 'Salary Expectation']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Group the data by 'Department', 'Gender', and 'College Mark', and calculate the average 'Salary Expectations'
grouped_data = df.groupby(['Department', 'Gender', '12th Mark'])['Salary Expectation'].mean().reset_index()

# Plotting using seaborn
plt.figure(figsize=(12, 8))
sns.barplot(data=grouped_data, x='Department', y='Salary Expectation', hue='Gender', ci=None)

# Set plot labels and title
plt.title('Salary Expectations by Department, Gender, and College Mark')
plt.xlabel('Department')
plt.ylabel('Average Salary Expectations')

# Show plot
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# In[9]:


pd.set_option('display.max_rows',None)
df= pd.read_csv(r"C:\Users\Noel\Downloads\Pandas\Student Attitude and Behavior.csv")
df


# In[7]:


# Sample data creation for illustration purposes
data = {
    'Social Media': ['0-1 hours', '1-2 hours', '2-3 hours', '3-4 hours', '4+ hours'],
    '10th Mark': [85, 80, 75, 70, 65],
    '12th Mark': [88, 82, 78, 72, 68],
    'College Mark': [90, 85, 80, 75, 70]
}
df = pd.DataFrame(data)

# If you have your data in a CSV file, use the following line instead
# df = pd.read_csv('path_to_your_file.csv')

# Ensure the relevant columns exist in the dataframe
required_columns = ['Social Media', '10th Mark', '12th Mark', 'College Mark']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Replace missing values with 0 in the academic performance columns
df['10th Mark'].fillna(0, inplace=True)
df['12th Mark'].fillna(0, inplace=True)
df['College Mark'].fillna(0, inplace=True)

# Group the data by Social Media Usage and calculate the average marks for each group
grouped_data = df.groupby('Social Media').agg({
    '10th Mark': 'mean',
    '12th Mark': 'mean',
    'College Mark': 'mean'
}).reset_index()

# Plotting
fig, ax = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

# Plot for 10th Mark
ax[0].bar(grouped_data['Social Media'], grouped_data['10th Mark'], color='skyblue')
ax[0].set_title('Average 10th Mark by Social Media Usage')
ax[0].set_ylabel('Average 10th Mark')

# Plot for 12th Mark
ax[1].bar(grouped_data['Social Media'], grouped_data['12th Mark'], color='lightgreen')
ax[1].set_title('Average 12th Mark by Social Media Usage')
ax[1].set_ylabel('Average 12th Mark')

# Plot for College Mark
ax[2].bar(grouped_data['Social Media'], grouped_data['College Mark'], color='salmon')
ax[2].set_title('Average College Mark by Social Media Usage')
ax[2].set_ylabel('Average College Mark')

# Set common x-label
ax[2].set_xlabel('Social Media Usage (hours per day)')

# Rotate x-ticks for better readability
plt.xticks(rotation=45)

# Adjust layout for better spacing
plt.tight_layout()

# Show plot
plt.show()


# In[10]:



# Sample data creation for illustration purposes
data = {
    'Side gig': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
    'Financial Status': ['Bad', 'Good', 'Awful', 'Fabulous', 'Good', 'Bad', 'Fabulous', 'Awful'],
    '10th Mark': [75, 85, 65, 90, 80, 70, 95, 60],
    '12th Mark': [78, 88, 68, 92, 82, 72, 98, 62],
    'College Mark': [80, 90, 70, 94, 85, 75, 99, 65]
}
df = pd.DataFrame(data)

# If you have your data in a CSV file, use the following line instead
# df = pd.read_csv('path_to_your_file.csv')

# Ensure the relevant columns exist in the dataframe
required_columns = ['Side gig', 'Financial Status', '10th Mark', '12th Mark', 'College Mark']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Replace missing values with 0 in the academic performance columns
df['10th Mark'].fillna(0, inplace=True)
df['12th Mark'].fillna(0, inplace=True)
df['College Mark'].fillna(0, inplace=True)

# Group the data by Side gig and Financial Status and calculate the average marks for each group
grouped_data = df.groupby(['Side gig', 'Financial Status']).agg({
    '10th Mark': 'mean',
    '12th Mark': 'mean',
    'College Mark': 'mean'
}).reset_index()

# Plotting
fig, ax = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

# Plot for 10th Mark
sns.barplot(data=grouped_data, x='Financial Status', y='10th Mark', hue='Side gig', ax=ax[0])
ax[0].set_title('Average 10th Mark by Side gig and Financial Status')
ax[0].set_ylabel('Average 10th Mark')

# Plot for 12th Mark
sns.barplot(data=grouped_data, x='Financial Status', y='12th Mark', hue='Side gig', ax=ax[1])
ax[1].set_title('Average 12th Mark by Side gig and Financial Status')
ax[1].set_ylabel('Average 12th Mark')

# Plot for College Mark
sns.barplot(data=grouped_data, x='Financial Status', y='College Mark', hue='Side gig', ax=ax[2])
ax[2].set_title('Average College Mark by Side gig and Financial Status')
ax[2].set_ylabel('Average College Mark')

# Set common x-label
ax[2].set_xlabel('Financial Status')

# Rotate x-ticks for better readability
plt.xticks(rotation=45)

# Adjust layout for better spacing
plt.tight_layout()

# Show plot
plt.show()


# In[11]:


# Sample data creation for illustration purposes
data = {
    'Salary Expectation': [50000, 60000, 55000, 70000, 62000, 58000, 68000, 75000],
    'Financial Status': ['Bad', 'Good', 'Awful', 'Fabulous', 'Good', 'Bad', 'Fabulous', 'Awful'],
    '10th Mark': [75, 85, 65, 90, 80, 70, 95, 60],
    '12th Mark': [78, 88, 68, 92, 82, 72, 98, 62],
    'College Mark': [80, 90, 70, 94, 85, 75, 99, 65]
}
df = pd.DataFrame(data)

# If you have your data in a CSV file, use the following line instead
# df = pd.read_csv('path_to_your_file.csv')

# Ensure the relevant columns exist in the dataframe
required_columns = ['Salary Expectation', 'Financial Status', '10th Mark', '12th Mark', 'College Mark']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Replace missing values with 0 in the academic performance columns
df['10th Mark'].fillna(0, inplace=True)
df['12th Mark'].fillna(0, inplace=True)
df['College Mark'].fillna(0, inplace=True)

# Group the data by Financial Status and calculate the average marks and salary expectations for each group
grouped_data = df.groupby('Financial Status').agg({
    'Salary Expectation': 'mean',
    '10th Mark': 'mean',
    '12th Mark': 'mean',
    'College Mark': 'mean'
}).reset_index()

# Plotting
fig, ax = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

# Plot for 10th Mark vs Salary Expectations
sns.barplot(data=grouped_data, x='Financial Status', y='Salary Expectation', hue='10th Mark', dodge=False, ax=ax[0])
ax[0].set_title('Average Salary Expectations by Financial Status and 10th Mark')
ax[0].set_ylabel('Average Salary Expectations')

# Plot for 12th Mark vs Salary Expectations
sns.barplot(data=grouped_data, x='Financial Status', y='Salary Expectation', hue='12th Mark', dodge=False, ax=ax[1])
ax[1].set_title('Average Salary Expectations by Financial Status and 12th Mark')
ax[1].set_ylabel('Average Salary Expectations')

# Plot for College Mark vs Salary Expectations
sns.barplot(data=grouped_data, x='Financial Status', y='Salary Expectation', hue='College Mark', dodge=False, ax=ax[2])
ax[2].set_title('Average Salary Expectations by Financial Status and College Mark')
ax[2].set_ylabel('Average Salary Expectations')

# Set common x-label
ax[2].set_xlabel('Financial Status')

# Rotate x-ticks for better readability
plt.xticks(rotation=45)

# Adjust layout for better spacing
plt.tight_layout()

# Show plot
plt.show()


# In[12]:


# Sample data creation for illustration purposes
data = {
    'Travelling Time': ['0-1 hour', '1-2 hours', '2-3 hours', '3-4 hours', '0-1 hour', '1-2 hours', '2-3 hours', '3-4 hours'],
    'Stress Levels': ['Good', 'Bad', 'Awful', 'fabulous', 'Good', 'Bad', 'Awful', 'fabulous'],
    '10th Mark': [75, 85, 65, 90, 80, 70, 95, 60],
    '12th Mark': [78, 88, 68, 92, 82, 72, 98, 62],
    'College Mark': [80, 90, 70, 94, 85, 75, 99, 65]
}
df = pd.DataFrame(data)

# If you have your data in a CSV file, use the following line instead
# df = pd.read_csv('path_to_your_file.csv')

# Ensure the relevant columns exist in the dataframe
required_columns = ['Travelling Time', 'Stress Levels', '10th Mark', '12th Mark', 'College Mark']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Replace missing values with 0 in the academic performance columns
df['10th Mark'].fillna(0, inplace=True)
df['12th Mark'].fillna(0, inplace=True)
df['College Mark'].fillna(0, inplace=True)

# Map Stress Levels to numeric values
stress_mapping = {'Awful': 0, 'Bad': 1, 'Good': 2, 'fabulous': 3}
df['Stress Levels'] = df['Stress Levels'].map(stress_mapping)

# Ensure the mapping was successful
if df['Stress Levels'].isnull().any():
    raise ValueError("There are values in 'Stress Levels' that were not mapped successfully.")

# Group the data by Travelling Time and calculate the average marks and stress levels for each group
grouped_data = df.groupby('Travelling Time').agg({
    'Stress Levels': 'mean',
    '10th Mark': 'mean',
    '12th Mark': 'mean',
    'College Mark': 'mean'
}).reset_index()

# Plotting
fig, axes = plt.subplots(4, 1, figsize=(15, 20))

# Plot for 10th Mark vs Travelling Time
sns.barplot(data=grouped_data, x='Travelling Time', y='10th Mark', ax=axes[0])
axes[0].set_title('Average 10th Mark by Travelling Time')
axes[0].set_ylabel('Average 10th Mark')

# Plot for 12th Mark vs Travelling Time
sns.barplot(data=grouped_data, x='Travelling Time', y='12th Mark', ax=axes[1])
axes[1].set_title('Average 12th Mark by Travelling Time')
axes[1].set_ylabel('Average 12th Mark')

# Plot for College Mark vs Travelling Time
sns.barplot(data=grouped_data, x='Travelling Time', y='College Mark', ax=axes[2])
axes[2].set_title('Average College Mark by Travelling Time')
axes[2].set_ylabel('Average College Mark')

# Scatter plot for Stress Levels vs Travelling Time
sns.scatterplot(data=grouped_data, x='Travelling Time', y='Stress Levels', size='Stress Levels', ax=axes[3])
axes[3].set_title('Average Stress Levels by Travelling Time')
axes[3].set_ylabel('Average Stress Levels')

# Rotate x-ticks for better readability
plt.xticks(rotation=45)

# Adjust layout for better spacing
plt.tight_layout()

# Show plot
plt.show()


# In[13]:


# Sample data creation for illustration purposes
data = {
    'Travelling Time': ['<15 mins', '15-30 mins', '30-60 mins', '>60 mins', '<15 mins', '15-30 mins', '30-60 mins', '>60 mins'],
    '10th Mark': [75, 85, 65, 90, 80, 70, 95, 60],
    '12th Mark': [78, 88, 68, 92, 82, 72, 98, 62],
    'College Mark': [80, 90, 70, 94, 85, 75, 99, 65]
}
df = pd.DataFrame(data)

# If you have your data in a CSV file, use the following line instead
# df = pd.read_csv('path_to_your_file.csv')

# Ensure the relevant columns exist in the dataframe
required_columns = ['Travelling Time', '10th Mark', '12th Mark', 'College Mark']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Replace missing values with 0 in the academic performance columns
df['10th Mark'].fillna(0, inplace=True)
df['12th Mark'].fillna(0, inplace=True)
df['College Mark'].fillna(0, inplace=True)

# Group the data by Travelling Time and calculate the average marks for each group
grouped_data = df.groupby('Travelling Time').agg({
    '10th Mark': 'mean',
    '12th Mark': 'mean',
    'College Mark': 'mean'
}).reset_index()

# Plotting
fig, ax = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

# Plot for 10th Mark vs Travelling Time
sns.barplot(data=grouped_data, x='Travelling Time', y='10th Mark', ax=ax[0])
ax[0].set_title('Average 10th Mark by Travelling Time')
ax[0].set_ylabel('Average 10th Mark')

# Plot for 12th Mark vs Travelling Time
sns.barplot(data=grouped_data, x='Travelling Time', y='12th Mark', ax=ax[1])
ax[1].set_title('Average 12th Mark by Travelling Time')
ax[1].set_ylabel('Average 12th Mark')

# Plot for College Mark vs Travelling Time
sns.barplot(data=grouped_data, x='Travelling Time', y='College Mark', ax=ax[2])
ax[2].set_title('Average College Mark by Travelling Time')
ax[2].set_ylabel('Average College Mark')

# Set common x-label
ax[2].set_xlabel('Travelling Time')

# Rotate x-ticks for better readability
plt.xticks(rotation=45)

# Adjust layout for better spacing
plt.tight_layout()

# Show plot
plt.show()


# In[15]:


# Sample data creation for illustration purposes
data = {
    'Side gig': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
    'Financial Status': ['Bad', 'good', 'Awful', 'Fabulous', 'Bad', 'good', 'Awful', 'Fabulous'],
    'Stress Levels': ['Awful', 'Bad', 'Good', 'fabulous', 'Awful', 'Bad', 'Good', 'fabulous'],
    '10th Mark': [75, 85, 65, 90, 80, 70, 95, 60],
    '12th Mark': [78, 88, 68, 92, 82, 72, 98, 62],
    'College Mark': [80, 90, 70, 94, 85, 75, 99, 65]
}
df = pd.DataFrame(data)

# If you have your data in a CSV file, use the following line instead
# df = pd.read_csv('path_to_your_file.csv')

# Ensure the relevant columns exist in the dataframe
required_columns = ['Side gig', 'Financial Status', 'Stress Levels', '10th Mark', '12th Mark', 'College Mark']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Replace missing values with 0 in the academic performance columns
df['10th Mark'].fillna(0, inplace=True)
df['12th Mark'].fillna(0, inplace=True)
df['College Mark'].fillna(0, inplace=True)

# Encode categorical variables for Financial Status and Stress Levels
financial_status_mapping = {'Awful': 0, 'Bad': 1, 'good': 2, 'Fabulous': 3}
stress_levels_mapping = {'Awful': 0, 'Bad': 1, 'Good': 2, 'fabulous': 3}

df['Financial Status'] = df['Financial Status'].map(financial_status_mapping)
df['Stress Levels'] = df['Stress Levels'].map(stress_levels_mapping)

# Group the data by Side Gig, Financial Status, and Stress Levels
grouped_data = df.groupby(['Side gig', 'Financial Status', 'Stress Levels']).agg({
    '10th Mark': 'mean',
    '12th Mark': 'mean',
    'College Mark': 'mean'
}).reset_index()

# Plotting
fig, ax = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

# Plot for 10th Mark
sns.barplot(data=grouped_data, x='Financial Status', y='10th Mark', hue='Side gig', ax=ax[0])
ax[0].set_title('Average 10th Mark by Financial Status, Stress Levels, and Side gig')
ax[0].set_ylabel('Average 10th Mark')

# Plot for 12th Mark
sns.barplot(data=grouped_data, x='Financial Status', y='12th Mark', hue='Side gig', ax=ax[1])
ax[1].set_title('Average 12th Mark by Financial Status, Stress Levels, and Side gig')
ax[1].set_ylabel('Average 12th Mark')

# Plot for College Mark
sns.barplot(data=grouped_data, x='Financial Status', y='College Mark', hue='Side gig', ax=ax[2])
ax[2].set_title('Average College Mark by Financial Status, Stress Levels, and Side gig')
ax[2].set_ylabel('Average College Mark')

# Set common x-label and x-ticks
ax[2].set_xlabel('Financial Status (0=Awful, 1=Bad, 2=good, 3=Fabulous)')
plt.xticks(rotation=45)

# Adjust layout for better spacing
plt.tight_layout()

# Show plot
plt.show()


# In[20]:



# Sample data creation for illustration purposes
data = {
    'Stress Levels': ['Awful', 'Bad', 'Good', 'Fabulous', 'Awful', 'Bad', 'Good', 'Fabulous'],
    'Side gig': ['Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
    '10th Mark': [75, 85, 65, 90, 80, 70, 95, 60],
    '12th Mark': [78, 88, 68, 92, 82, 72, 98, 62],
    'College Mark': [80, 90, 70, 94, 85, 75, 99, 65]
}
df = pd.DataFrame(data)

# If you have your data in a CSV file, use the following line instead
# df = pd.read_csv('path_to_your_file.csv')

# Ensure the relevant columns exist in the dataframe
required_columns = ['Stress Levels', 'Side gig', '10th Mark', '12th Mark', 'College Mark']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Convert categorical values in 'Stress Levels' to numerical values
stress_levels_mapping = {'Awful': 0, 'Bad': 1, 'Good': 2, 'Fabulous': 3}
df['Stress Levels'] = df['Stress Levels'].map(stress_levels_mapping)

# Plotting scatter plots for each academic performance category against Stress Levels
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot for 10th Mark vs Stress Levels
sns.scatterplot(data=df, x='Stress Levels', y='10th Mark', hue='Side gig', ax=axs[0])
axs[0].set_title('10th Mark vs Stress Levels')

# Plot for 12th Mark vs Stress Levels
sns.scatterplot(data=df, x='Stress Levels', y='12th Mark', hue='Side gig', ax=axs[1])
axs[1].set_title('12th Mark vs Stress Levels')

# Plot for College Mark vs Stress Levels
sns.scatterplot(data=df, x='Stress Levels', y='College Mark', hue='Side gig', ax=axs[2])
axs[2].set_title('College Mark vs Stress Levels')

# Show plot
plt.tight_layout()
plt.show()


# In[24]:


# Sample data creation for illustration purposes
data = {
    'Daily Study Time': ['0 - 30 minutes', '30 - 60 minutes', '1 - 2 hours', '2 - 3 hours'],
    'Stress Levels': ['Awful', 'Bad', 'Good', 'Fabulous'],
    '10th Mark': [75, 85, 65, 90],
    '12th Mark': [78, 88, 68, 92],
    'College Mark': [80, 90, 70, 94]
}
df = pd.DataFrame(data)

# If you have your data in a CSV file, use the following line instead
# df = pd.read_csv('path_to_your_file.csv')

# Ensure the relevant columns exist in the dataframe
required_columns = ['Daily Study Time', 'Stress Levels', '10th Mark', '12th Mark', 'College Mark']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Convert categorical values in 'Stress Levels' to numerical values
stress_levels_mapping = {'Awful': 0, 'Bad': 1, 'Good': 2, 'Fabulous': 3}
df['Stress Levels'] = df['Stress Levels'].map(stress_levels_mapping)

# Plotting scatter plots for each academic performance category against Daily Study Time and Stress Levels
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot for 10th Mark vs Daily Study Time and Stress Levels
sns.scatterplot(data=df, x='Daily Study Time', y='10th Mark', hue='Stress Levels', ax=axs[0])
axs[0].set_title('10th Mark vs Daily Study Time and Stress Levels')

# Plot for 12th Mark vs Daily Study Time and Stress Levels
sns.scatterplot(data=df, x='Daily Study Time', y='12th Mark', hue='Stress Levels', ax=axs[1])
axs[1].set_title('12th Mark vs Daily Study Time and Stress Levels')

# Plot for College Mark vs Daily Study Time and Stress Levels
sns.scatterplot(data=df, x='Daily Study Time', y='College Mark', hue='Stress Levels', ax=axs[2])
axs[2].set_title('College Mark vs Daily Study Time and Stress Levels')

# Show plot
plt.tight_layout()
plt.show()


# In[27]:


# Sample data creation for illustration purposes
data = {
    'Daily Study Time': ['0-2 hours', '0-2 hours', '2-4 hours', '2-4 hours', '4-6 hours', '4-6 hours', '6+ hours', '6+ hours'],
    'Stress Levels': ['Awful', 'Bad', 'Good', 'Fabulous', 'Awful', 'Bad', 'Good', 'Fabulous'],
    '10th Mark': [75, 85, 65, 90, 80, 70, 95, 60],
    '12th Mark': [78, 88, 68, 92, 82, 72, 98, 62],
    'College Mark': [80, 90, 70, 94, 85, 75, 99, 65]
}
df = pd.DataFrame(data)

# If you have your data in a CSV file, use the following line instead
# df = pd.read_csv('path_to_your_file.csv')

# Ensure the relevant columns exist in the dataframe
required_columns = ['Daily Study Time', 'Stress Levels', '10th Mark', '12th Mark', 'College Mark']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing from the dataset")

# Convert categorical values in 'Stress Levels' to numerical values
stress_levels_mapping = {'Awful': 0, 'Bad': 1, 'Good': 2, 'Fabulous': 3}
df['Stress Levels'] = df['Stress Levels'].map(stress_levels_mapping)

# Plotting grouped bar plots for each academic performance category against different levels of Daily Study Time and Stress Levels
academic_performance_columns = ['10th Mark', '12th Mark', 'College Mark']
for column in academic_performance_columns:
    fig, axs = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x='Daily Study Time', y=column, hue='Stress Levels', ax=axs)
    axs.set_title(f'{column} vs Daily Study Time & Stress Levels')
    axs.set_ylabel(column)
    axs.set_xlabel('Daily Study Time')
    plt.legend(title='Stress Levels')
    plt.show()


# In[ ]:




