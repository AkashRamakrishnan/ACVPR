import os
import json

def merge_json_files(directory, output_file):
    merged_data = {}

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)

            # Load the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Merge the dictionaries
            merged_data.update(data)

    # Write the merged data to the output file
    with open(output_file, 'w') as file:
        json.dump(merged_data, file)

    print(f"Merged data written to {output_file}.")

# Provide the directory path and output file name
directory_path = 'json'
output_file = 'merged_data.json'

# Call the function to merge the JSON files
merge_json_files(directory_path, output_file)
