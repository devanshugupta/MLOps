import pandas as pd
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport
import os

# Step 1: Load the Dataset
file_path = "abalone/abalone.data"  # Replace with your .data file path
df = pd.read_csv(file_path, header=None)

# Step 2: Dataset Schema and Storage
# Define the dataset schema
schema = {
    "Variable Name": [
        "Sex", "Length", "Diameter", "Height", "Whole_weight",
        "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"
    ],
    "Role": [
        "Feature", "Feature", "Feature", "Feature", "Feature",
        "Feature", "Feature", "Feature", "Target"
    ],
    "Type": [
        "Categorical", "Continuous", "Continuous", "Continuous", "Continuous",
        "Continuous", "Continuous", "Continuous", "Integer"
    ],
    "Description": [
        "M, F, and I (infant)",
        "Longest shell measurement",
        "Perpendicular to length",
        "With meat in shell",
        "Whole abalone",
        "Weight of meat",
        "Gut weight (after bleeding)",
        "After being dried",
        "+1.5 gives the age in years"
    ],
    "Units": [
        "", "mm", "mm", "mm", "grams",
        "grams", "grams", "grams", ""
    ],
    "Missing Values": ["no", "no", "no", "no", "no", "no", "no", "no", "no"]
}
schema_df = pd.DataFrame(schema)
print("Dataset Schema:")
print(schema_df)

# Save the schema to a Parquet file
schema_df.to_parquet("abalone_dataset_schema.parquet", index=False)

# Save the full dataset as Parquet
df.to_parquet("abalone_full_dataset.parquet", index=False)

# Step 3: Profiling the Dataset
profile = ProfileReport(df, title="Abalone Dataset Profile", explorative=True)
profile.to_file("abalone_dataset_profile.html")  # Generates a detailed profiling report

# Step 4: Train-Test Split
# Perform the split
train, temp = train_test_split(df, test_size=0.4, random_state=42)  # 60% train, 40% for test+production
test, production = train_test_split(temp, test_size=0.5, random_state=42)  # Split test and production equally

# Save splits as Parquet
os.makedirs("abalone_splits", exist_ok=True)
train.to_parquet("abalone_splits/train.parquet", index=False)
test.to_parquet("abalone_splits/test.parquet", index=False)
production.to_parquet("abalone_splits/production.parquet", index=False)

print("Abalone data processing completed successfully.")
