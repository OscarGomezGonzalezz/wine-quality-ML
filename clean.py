import pandas as pd
import numpy as np
import os

def clean_dataset(csv_file_path, output_folder="cleanDatasets"):
    # Load the dataset from a CSV file (local or remote)
    df = pd.read_csv(csv_file_path, sep=";", quotechar='"')
    
    # Initial inspection
    print("First rows of the dataset:")
    print(df.head())

    # General data inspection
    print("\nGeneral information of the DataFrame:")
    print(df.info())
    
    print("\nDescriptive statistics:")
    print(df.describe())
    
    print("\nMissing values by column:")
    print(df.isnull().sum())

    # Data cleaning
    print("\nRemoving duplicates and filling missing values...")
    # Remove duplicate rows if they exist
    df = df.drop_duplicates()

    # Fill missing values with the mean (only for numeric columns)
    df = df.fillna(df.mean(numeric_only=True))

    # Final verification
    print("\nPreview after cleaning and transformation:")
    print(df.head())
    
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    # Create the folder to save cleaned datasets if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create a filename for the cleaned file using the original name
    output_filename = os.path.basename(csv_file_path).replace('.csv', '_cleaned.csv')

    # Save the cleaned dataset
    output_path = os.path.join(output_folder, output_filename)
    df.to_csv(output_path, index=False)
    print(f"\n✔️ Cleaned dataset saved as '{output_path}'")
    return df



