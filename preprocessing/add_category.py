import pandas as pd

# Load the original dataset
file_path = '/home/ubuntu/reflexion/New_LLM_Agent_4_Ad_Keyword_and_Text/preprocessing/data/Search keyword report.csv'
data = pd.read_csv(file_path, encoding='utf-16', delimiter='\t')

# Adjust the headers and clean up the data
new_header = data.iloc[1]  # Use the second row as the header
data = data[2:]  # Take the data excluding the header row
data.columns = new_header  # Set the header row as the dataframe header
data.reset_index(drop=True, inplace=True)

# Print the first few rows of the original data
print("Original Data Keywords:")
print(data['Keyword'].head())

# Load the category mapping file
file_path_categories = '/home/ubuntu/reflexion/New_LLM_Agent_4_Ad_Keyword_and_Text/preprocessing/data/Initial_KW.csv'
category_data = pd.read_csv(file_path_categories)

# Reshape the category data for easy merging
melted_category_data = category_data.melt(id_vars=[category_data.columns[0]], var_name='column', value_name='Keyword')
melted_category_data = melted_category_data[['Keyword', category_data.columns[0]]]  # Keep only the relevant columns
melted_category_data.columns = ['Keyword', 'Category']  # Rename columns for clarity
melted_category_data.dropna(inplace=True)  # Remove any rows with NaN values


# Print the first few rows of the category data
print("Category Data Keywords:")
print(melted_category_data['Keyword'].head())

# Merge the original data with the category data
merged_data = pd.merge(data, melted_category_data, on='Keyword', how='left')
merged_data['Category'] = merged_data['Category_y'].combine_first(merged_data['Category_x'])
merged_data.drop(columns=['Category_x', 'Category_y'], inplace=True)
# Remove the last two rows which are not needed
merged_data = merged_data[:-2]

# Print the first few rows of the merged data
print("Merged Data Sample:")
print(merged_data.head())

# Save the updated dataset to a new file
output_file_path = '/home/ubuntu/reflexion/New_LLM_Agent_4_Ad_Keyword_and_Text/preprocessing/data/updated_keyword_report.csv'
merged_data.to_csv(output_file_path, index=False)

print("Updated dataset saved to:", output_file_path)