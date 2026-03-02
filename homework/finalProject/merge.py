import pandas as pd

# 1. Load the two CSV files
# Replace 'file1.csv' and 'file2.csv' with your actual file paths
df1 = pd.read_csv('/Users/y/Desktop/344MLhomework/homework/finalProject/landmarks_data.csv')
df2 = pd.read_csv('/Users/y/Desktop/344MLhomework/homework/finalProject/landmarks_data_right.csv')

# 2. Add the 'direction' column to each DataFrame
df1['direction'] = 'Pointing_Left'
df2['direction'] = 'Pointing_Right'

# 3. Merge both DataFrames into one
# ignore_index=True ensures the new DataFrame has a continuous index
combined_df = pd.concat([df1, df2], ignore_index=True)

# 4. Mix up (shuffle) the rows
# frac=1 means it samples 100% of the rows. 
# random_state ensures reproducibility (optional, remove if you want a different shuffle every time).
# reset_index cleans up the row numbers after shuffling.
shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Drop certain column
shuffled_df = shuffled_df.drop(columns=['hand_label', 'score'])

# 5. Export to a new huge CSV file
shuffled_df.to_csv('merged_dataset.csv', index=False)

print("Files successfully merged and shuffled!")