#!/bin/bash
start_time=$(date +%s) 

# Set the input CSV file and the Python script
csv_file="sents.csv"
python_script="roundtrip.py"

line_number=0

# Read lines from CSV file
while IFS=, read -r line; do
    ((line_number++))
    echo "line: $line_number"
    python "$python_script" "$line"
    
done < "$csv_file"
end_time=$(date +%s)  


elapsed_time=$((end_time - start_time))
echo "Script execution time: $elapsed_time seconds"