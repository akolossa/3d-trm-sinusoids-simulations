unique_combinations = {}
total_counts = {}

with open('/home/arawa/tst/optimal_parameters_100sim.BO.txt', 'r') as file:
    for line in file:
        parts = line.split()
        key = ' '.join(parts[:6])
        if key not in total_counts:
            total_counts[key] = 0
        total_counts[key] += 1
        
        if parts[7] == 'False' and parts[8] == '49':
            if key not in unique_combinations:
                unique_combinations[key] = {'total': 0, 'below_5': 0, 'above_30': 0, 'in_between5_30_exclusive': 0}
            unique_combinations[key]['total'] += 1
            if float(parts[6]) <= 5:
                unique_combinations[key]['below_5'] += 1
            if float(parts[6]) >= 30:
                unique_combinations[key]['above_30'] += 1
            if 5 < float(parts[6]) < 30:
                unique_combinations[key]['in_between5_30_exclusive'] += 1
                
with open('/home/arawa/tst/100sim_percentages_results.BO.txt', 'w') as output_file:
    for key, counts in unique_combinations.items():
        percentage = (counts['total'] / total_counts[key]) * 100
        output = (
            f"Parameters: {key}\n"
            f"Percentage of lines that are false and have 49 iterations: {percentage:.2f}%\n"
            "----\n"
            f"# lines of {percentage:.2f}% with displacement between 5 and 30 (excl): {counts['in_between5_30_exclusive']}\n"
            f"# lines of {percentage:.2f}% with displacement below 5 (incl): {counts['below_5']}\n"
            f"# lines of {percentage:.2f}% with displacement above 30 (incl): {counts['above_30']}\n"
            "----------\n"
        )
        print(output)
        output_file.write(output)

    