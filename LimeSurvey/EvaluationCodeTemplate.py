#Import Libraries and Packages
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive
import re
drive.mount('/content/drive')



#If you are using Google Colab
drive.mount('/content/drive')

file_path = '/content/drive/MyDrive/results-survey551398.csv'  
data = pd.read_csv(file_path)

print(data.head())



# Correcting the file path and reloading the survey dataset
file_path = '/content/drive/MyDrive/results-survey551398.csv'
survey_data = pd.read_csv(file_path)

# Strip extra spaces in column names
survey_data.columns = survey_data.columns.str.strip()

# Redefine demographic data
demographics_columns = [
    "Response ID",
    "To which age group do you belong?",
    "What is your gender?",
    "What is your nationality? Please select from the dropdown menu."
]
demographics_data = survey_data[demographics_columns]

# Generate summary
demographics_summary = demographics_data.groupby(
    ["To which age group do you belong?", "What is your gender?", "What is your nationality? Please select from the dropdown menu."]
).size().reset_index(name='Count')

# Visualize age group distribution
import matplotlib.pyplot as plt

demographics_data["To which age group do you belong?"].value_counts().plot(kind='bar')
plt.title("Age Group Distribution")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.show()

# Visualize gender distribution
demographics_data["What is your gender?"].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Gender Distribution")
plt.ylabel("")
plt.show()

# Visualize nationality distribution
demographics_data["What is your nationality? Please select from the dropdown menu."].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Nationality Distribution")
plt.ylabel("")
plt.show()

# Save demographic summary to a CSV file
output_file = "demographic_summary.csv"
demographics_summary.to_csv(output_file, index=False)
print(f"Demographic summary saved as '{output_file}'.")



# Load the CSV file
file_path = '/content/drive/MyDrive/results-survey551398.csv'
survey_data = pd.read_csv(file_path)

# Strip extra spaces in column names
survey_data.columns = survey_data.columns.str.strip()

# Define fields for educational background and level of education
education_fields = [
    "What is your primary field of study or educational background? [Natural Sciences and Mathematics]",
    "What is your primary field of study or educational background? [Engineering, Technology, and Computer Sciences]",
    "What is your primary field of study or educational background? [Medical, Health Sciences]",
    "What is your primary field of study or educational background? [Social Sciences and Humanities]",
    "What is your primary field of study or educational background? [Arts, Design, and Architecture]",
]

education_levels = [
    "What is the highest level of education you have completed?  [High school or equivalent]",
    "What is the highest level of education you have completed?  [Bachelor’s degree]",
    "What is the highest level of education you have completed?  [Master’s degree]",
    "What is the highest level of education you have completed?  [PHD]",
]

# Shorten the labels for visualization
shortened_fields = {
    education_fields[0]: "Natural Sciences and Mathematics",
    education_fields[1]: "Engineering, Technology, and Computer Sciences",
    education_fields[2]: "Medical, Health Sciences",
    education_fields[3]: "Social Sciences and Humanities",
    education_fields[4]: "Arts, Design, and Architecture",
}

shortened_levels = {
    education_levels[0]: "High School",
    education_levels[1]: "Bachelor's",
    education_levels[2]: "Master's",
    education_levels[3]: "PhD",
}

# Convert "Yes" to 1 and "No" to 0
education_fields_data = survey_data[education_fields].replace({"Yes": 1, "No": 0}).fillna(0)
education_levels_data = survey_data[education_levels].replace({"Yes": 1, "No": 0}).fillna(0)

# Rename the columns for simplicity
education_fields_data.rename(columns=shortened_fields, inplace=True)
education_levels_data.rename(columns=shortened_levels, inplace=True)

# Summarize educational fields
education_field_counts = education_fields_data.sum().reset_index()
education_field_counts.columns = ["Field of Study", "Count"]

# Summarize education levels
education_level_counts = education_levels_data.sum().reset_index()
education_level_counts.columns = ["Education Level", "Count"]

# Plot educational background (fields of study)
education_field_counts.set_index("Field of Study").plot(kind='bar', legend=False)
plt.title("Distribution of Fields of Study")
plt.xlabel("Field of Study")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Plot highest level of education
education_level_counts.set_index("Education Level").plot(kind='bar', legend=False)
plt.title("Distribution of Highest Level of Education")
plt.xlabel("Education Level")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Save summaries as CSV files for reference
education_field_counts.to_csv("field_of_study_summary.csv", index=False)
education_level_counts.to_csv("education_level_summary.csv", index=False)
print("Summaries saved as 'field_of_study_summary.csv' and 'education_level_summary.csv'")



# Re-define the tweet labels variable
tweet_labels = [col for col in survey_data.columns if "label" in col.lower()]

# Standardize the labels to ensure no duplicates
cleaned_labels = pd.Series(
    [label.split(":")[-1].strip().lower().replace("hatespeech", "hate speech") for label in tweet_labels]
).value_counts().reset_index()
cleaned_labels.columns = ["Category", "Count"]

# Plot the final corrected distribution of tweet categories
cleaned_labels.set_index("Category").plot(kind='bar', legend=False)
plt.title("Final Distribution of Tweet Categories")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Save the final cleaned summary as a CSV file
cleaned_labels.to_csv("final_tweet_category_summary.csv", index=False)
print("Final tweet category summary saved as 'final_tweet_category_summary.csv'.")



# Define columns for reasons behind explanation selection
explanation_reason_columns = [
    "After reviewing the explanations you selected, please tell us why you made your choice. You may select multiple reasons if applicable. [Clarity]",
    "After reviewing the explanations you selected, please tell us why you made your choice. You may select multiple reasons if applicable. [Plausibility]",
    "After reviewing the explanations you selected, please tell us why you made your choice. You may select multiple reasons if applicable. [Context]",
    "After reviewing the explanations you selected, please tell us why you made your choice. You may select multiple reasons if applicable. [Emotional Tone]",
    "After reviewing the explanations you selected, please tell us why you made your choice. You may select multiple reasons if applicable. [Fairness]",
]

# Shorten the column names for visualization
shortened_labels = {
    explanation_reason_columns[0]: "Clarity",
    explanation_reason_columns[1]: "Plausibility",
    explanation_reason_columns[2]: "Context",
    explanation_reason_columns[3]: "Emotional Tone",
    explanation_reason_columns[4]: "Fairness",
}

# Convert "Yes" to 1 and "No" to 0 for binary representation of reasons
explanation_reasons_data = survey_data[explanation_reason_columns].replace({"Yes": 1, "No": 0}).fillna(0)

# Rename the columns for simplicity
explanation_reasons_data.rename(columns=shortened_labels, inplace=True)

# Summarize reasons for selecting explanations
explanation_reason_counts = explanation_reasons_data.sum().reset_index()
explanation_reason_counts.columns = ["Reason", "Count"]

# Visualize reasons for selecting explanations
explanation_reason_counts.set_index("Reason").plot(kind='bar', legend=False)
plt.title("Reasons for Selecting Explanations")
plt.xlabel("Reason")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Save the summary of reasons as a CSV file
explanation_reason_counts.to_csv("explanation_reasons_summary.csv", index=False)
print("Explanations reasons summary saved as 'explanation_reasons_summary.csv'.")



# Load the uploaded CSV file
file_path = '/content/drive/MyDrive/statistic-survey551398.csv'

# Step 1: Open the file and extract relevant data
with open(file_path, 'r') as file:
    content = file.readlines()

# Step 2: Extract lines that contain explanation counts
cleaned_data = []
for line in content:
    if "Explanation 1:" in line or "Explanation 2:" in line or "Explanation 3:" in line:
        cleaned_data.append(line.strip())

# Step 3: Initialize a dictionary to store counts for each explanation type
explanation_counts = {"Zero Prompt": 0, "Few Shot": 0, "Chain of Thought": 0}
total_counts = 0

# Step 4: Parse cleaned data to extract counts for each explanation type
for line in cleaned_data:
    match = re.search(r'Explanation (\d):', line)
    if match:
        explanation_type = int(match.group(1))
        count_match = re.search(r';(\d+);', line)
        if count_match:
            count = int(count_match.group(1))
            if explanation_type == 1:
                explanation_counts["Zero Prompt"] += count
            elif explanation_type == 2:
                explanation_counts["Few Shot"] += count
            elif explanation_type == 3:
                explanation_counts["Chain of Thought"] += count
            total_counts += count

# Step 5: Calculate percentages for each explanation type
percentages = {
    key: (value / total_counts * 100) if total_counts > 0 else 0
    for key, value in explanation_counts.items()
}

# Step 6: Create a summary table
summary_table = pd.DataFrame({
    "Explanation Type": explanation_counts.keys(),
    "Number of Selections": explanation_counts.values(),
    "Percentage of Total": [f"{percent:.2f}%" for percent in percentages.values()]
})

# Print the table
print(summary_table)

# Step 7: Create a visualization
plt.figure(figsize=(8, 6))
plt.bar(explanation_counts.keys(), explanation_counts.values())
plt.title("Number of Selections by Explanation Type", fontsize=14)
plt.xlabel("Explanation Type", fontsize=12)
plt.ylabel("Number of Selections", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


