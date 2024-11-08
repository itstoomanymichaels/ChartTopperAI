import pandas as pd

# Load the CSV files into DataFrames
csv1 = pd.read_csv(r"C:\Users\tooma\OneDrive\ChartTopperAI\Merger\songs.csv", on_bad_lines='warn', encoding="latin-1")
csv2 = pd.read_csv(r"C:\Users\tooma\OneDrive\ChartTopperAI\Merger\TargetData.csv", on_bad_lines='warn', encoding="latin-1")

# Rename columns in csv2 to match the column names in csv1 for easy merging
csv2.rename(columns={'track': 'track_name'}, inplace=True)

# Perform an inner merge on 'track_name' and 'artist' to find matching rows
merged_df = pd.merge(csv1, csv2[['track_name', 'artist', 'target']], on=['track_name', 'artist'], how='inner')

# Save the merged data to a new CSV file
merged_df.to_csv('data.csv', index=False)

print("Merged CSV file has been created successfully.")