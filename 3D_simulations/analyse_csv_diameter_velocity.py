import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from parameters import parameters

# Load the CSV file
csv_file = f"/home/arawa/Shabaz_simulation/figure_7_ak/current_vti_files/output_{parameters['dimension']}_{parameters['version']}/simulation_metrics{parameters['dimension']}_{parameters['version']}.csv"
data = pd.read_csv(csv_file)

# Filter and clean data
# Convert columns to numeric if they contain list-like strings
data["velocity"] = pd.to_numeric(data["velocity"], errors="coerce")
#data["displacement"] = pd.to_numeric(data["displacement"], errors="coerce")
data["avg_displacement"] = pd.to_numeric(data["avg_displacement"], errors="coerce")
data = data.dropna()
data = data[data["cell_diameter"] > 0]
data = data[data["sinusoid_diameter"] > 0]

# Compute basic statistics
summary_stats = {
    "Mean Cell Diameter": data["cell_diameter"].mean(),
    "Std Cell Diameter": data["cell_diameter"].std(),
    "Mean Sinusoid Diameter": data["sinusoid_diameter"].mean(),
    "Std Sinusoid Diameter": data["sinusoid_diameter"].std(),
    "Mean Velocity": data["velocity"].mean(),
    "Std Velocity": data["velocity"].std(),
    "Mean Displacement": data["avg_displacement"].mean(),
    "Std Displacement": data["avg_displacement"].std(),
    "Mean Cell Elongation": data["cell_elongation"].mean(),
    "Std Cell Elongation": data["cell_elongation"].std(),
}

# Generate conclusions
conclusions = [
    f"The typical cell diameter is {summary_stats['Mean Cell Diameter']:.2f} ± {summary_stats['Std Cell Diameter']:.2f}.",
    f"The typical sinusoid diameter is {summary_stats['Mean Sinusoid Diameter']:.2f} ± {summary_stats['Std Sinusoid Diameter']:.2f}.",
    f"The mean cell velocity is {summary_stats['Mean Velocity']:.2f} ± {summary_stats['Std Velocity']:.2f}.",
    f"The mean cell displacement is {summary_stats['Mean Displacement']:.2f} ± {summary_stats['Std Displacement']:.2f}.",
    f"The mean cell elongation is {summary_stats['Mean Cell Elongation']:.2f} ± {summary_stats['Std Cell Elongation']:.2f}."
]

# Correlation analysis
correlation_matrix = data[["velocity", "avg_displacement", "cell_diameter", "sinusoid_diameter", "cell_elongation"]].corr()

# Print conclusions
print("\nConclusions:")
for conclusion in conclusions:
    print(f"- {conclusion}")

# Velocity distributions
plt.figure(figsize=(8, 6))
sns.histplot(data["velocity"], bins=20, kde=True)
plt.xlabel("Velocity")
plt.title("Velocity Distribution")
plt.show()

# Displacement distributions
plt.figure(figsize=(8, 6))
sns.histplot(data["avg_displacement"], bins=20, kde=True)
plt.xlabel("Displacement")
plt.title("Displacement Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Between Movement and Cell Properties")
plt.show()

# Effect of Elongation
plt.figure(figsize=(8, 6))
sns.scatterplot(x="cell_elongation", y="velocity", data=data)
plt.xlabel("Cell Elongation")
plt.ylabel("Velocity")
plt.title("Effect of Cell Elongation on Velocity")
plt.show()

# Effect of Contact Area on Movement
plt.figure(figsize=(8, 6))
sns.scatterplot(x="contact_area_with_sinusoids", y="velocity", data=data)
plt.xlabel("Contact Area with Sinusoids")
plt.ylabel("Velocity")
plt.title("Effect of Contact Area on Cell Movement")
plt.show()

# Time series analysis
plt.figure(figsize=(8, 6))
sns.lineplot(x="timepoint", y="velocity", data=data, marker="o")
plt.xlabel("Timepoint")
plt.ylabel("Velocity")
plt.title("Velocity Over Time")
plt.show()

# Save summary stats and conclusions
with open(f"analysis_summary{parameters['dimension']}_{parameters['version']}.txt", "w") as f:
    f.write("Summary Statistics:\n")
    for key, value in summary_stats.items():
        f.write(f"{key}: {value}\n")
    f.write("\nConclusions:\n")
    for conclusion in conclusions:
        f.write(f"- {conclusion}\n")
    f.write("\nCorrelation Matrix:\n")
    f.write(correlation_matrix.to_string())
    
print("Analysis completed. Summary saved.")
