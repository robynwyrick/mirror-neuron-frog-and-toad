#!/usr/bin/env python

import pandas as pd
import os
import zipfile

# Define file paths
zip_file = '../3_Labeling_Step/labeled_game_states.zip'
extracted_file = '../3_Labeling_Step/labeled_game_states.csv'
test_file = 'new_labeled_game_states_test.csv'
train_file = 'new_labeled_game_states_train.csv'
test_scenario_file = 'new_labeled_game_states_test.scenario.csv'

# Extract the labeled game states file
def extract_zip(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(extract_to))

# Check if the extracted file exists; if not, extract it
if not os.path.exists(extracted_file):
    extract_zip(zip_file, extracted_file)

# Read the labeled game states without headers
df = pd.read_csv(extracted_file, header=None)

# Manually assign column names
num_columns = df.shape[1]
column_names = ['col_{}'.format(i) for i in range(num_columns - 1)] + ['label']
df.columns = column_names

# Define the total number of samples for the test dataset
total_test_samples = 100000  # Set your desired total sample size here

# Define sampling ratios
ratios = {
    0: 0.4,  # 40% for label 0
    1: 0.4,  # 40% for label 1
    2: 0.1,  # 10% for label 2
    3: 0.1   # 10% for label 3
}

# Calculate the number of samples for each label based on the total size and ratios
sample_size_0 = int(total_test_samples * ratios[0])
sample_size_1 = int(total_test_samples * ratios[1])
sample_size_2 = int(total_test_samples * ratios[2])
sample_size_3 = int(total_test_samples * ratios[3])

# Draw samples for each label based on calculated sample sizes
df_0 = df[df['label'] == 0].sample(n=sample_size_0)
df_1 = df[df['label'] == 1].sample(n=sample_size_1)
df_2 = df[df['label'] == 2].sample(n=sample_size_2)
df_3 = df[df['label'] == 3].sample(n=sample_size_3)

# Concatenate to form the test dataset
df_test = pd.concat([df_0, df_1, df_2, df_3]).sample(frac=1).reset_index(drop=True)

# Save a copy of the test dataset before zeroing out column 98
df_test.to_csv(test_scenario_file, index=False, header=False)

# Remove test dataset from the original dataframe to create the train dataset
df_train = df.drop(df_test.index)

# Zero out column 'col_98' in df_test and df_train
df_test['col_98'] = 0
df_train['col_98'] = 0

# Save the test dataset
df_test.to_csv(test_file, index=False, header=False)

# Save the train dataset
df_train.to_csv(train_file, index=False, header=False)

print(f'Test dataset saved to {test_file}')
print(f'Train dataset saved to {train_file}')
print(f'Copy of test dataset saved to {test_scenario_file}')

