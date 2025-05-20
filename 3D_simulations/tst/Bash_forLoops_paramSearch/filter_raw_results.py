from collections import defaultdict

input_file = '/home/arawa/tst/output_v2_AK.txt'
output_file = '/home/arawa/tst/filtered_settings_movAbove5_3lines.v3.txt'

#filter based on the following conditions
# 1. movement between certain values
# 2. cell is not broken
# 3. iterations are 49
# 4. 3 lines with same first 5 parts
# Read lines from the input file and filter based on conditions
filtered_lines = []
with open(input_file, 'r') as infile:
    for line in infile:
        parts = line.split()
        broken_is_false = parts[6].lower() == 'false'  # checks if cell is not broken
        if float(parts[5]) > 5 and broken_is_false and float(parts[7]) == 49:  # if cell is moving and not broken
            filtered_lines.append(line)

# Dictionary to count occurrences of lines with the same first 5 parts
line_counts = defaultdict(int)
line_dict = defaultdict(list)

# Process each filtered line
for line in filtered_lines:
    parts = line.split()
    if len(parts) >= 5:
        key = tuple(parts[:5])
        line_counts[key] += 1
        line_dict[key].append(line)

# Write lines to the output file only if there are exactly 2 or 3 lines with the same first 5 parts
with open(output_file, 'w') as file:
    for key, count in line_counts.items():
        if count == 3: #2 or 3 lines with the same first 5 parts
            for line in line_dict[key]:
                file.write(line)