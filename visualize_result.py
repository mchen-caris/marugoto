import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt

# Set the path to your results directory
results_dir = './results'

# Initialize a dictionary to store AUC values per folder category
auc_dict = defaultdict(list)

# Traverse the directory tree
for root, _, files in os.walk(results_dir):
    for file in files:
        if file == 'roc-Malign=1.svg':
            folder_name = os.path.basename(root)
            folder_category = '_'.join(folder_name.split('_')[:-1])

            # Read AUC value from the SVG file
            with open(os.path.join(root, file), 'r') as f:
                content = f.read()
                auc_match = re.search(r'<!-- \(AUC = (\d+\.\d+)\) -->', content)
                if auc_match:
                    auc_value = float(auc_match.group(1))
                    auc_dict[folder_category].append(auc_value)
                    print(auc_value)

# Create box chart from AUC values
categories = list(auc_dict.keys())
auc_values = [auc_dict[category] for category in categories]

fig, ax = plt.subplots()
ax.boxplot(auc_values, labels=categories)
ax.set_title('AUC values per folder category')
ax.set_xlabel('Folder category')
ax.set_ylabel('AUC value')

# Save the plot as an image file
plt.savefig('box_plot.png', dpi=300)

plt.show()