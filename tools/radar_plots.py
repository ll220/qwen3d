import numpy as np
import matplotlib.pyplot as plt
from math import pi

import ipdb
st = ipdb.set_trace

# Example data
categories = ['3D Inst. Seg.', '3D Ref Grnd.', '3D QA', '2D Ref Grd.', '2D QA']
values_1 = np.array([12.1, 58.1, 58.6, 90.7, 60.0])
values_2 = np.array([16, 63.4, 58.8, 89.8, 59.8])
# max_values = np.array([36, 75, 81, 70, 30, 53])
percentage_increase = (values_2 - values_1) / values_1
base = percentage_increase.max()
outer_number_pos = [0.885, 0.76, 0.88, 0.85, 0.72, 0.60]

print(percentage_increase)
# print(base_scale)
# print(max_increase)

# Normalize values
# normalized_values_1 = [v / mv for v, mv in zip(percentage_increase, percentage_increase * 5.0)]
# normalized_values_2 = [v / mv for v, mv in zip(percentage_increase + percentage_increase.min(), percentage_increase * 5.0)]
normalized_values_1 = [v / mv for v, mv in zip(np.full(percentage_increase.shape, base), np.full(percentage_increase.shape, base * 2.5))]
normalized_values_2 = [v / mv for v, mv in zip(np.full(percentage_increase.shape, base + percentage_increase), np.full(percentage_increase.shape, base * 2.5))]
print(normalized_values_1)
print(normalized_values_2)

# Number of variables we're plotting.
num_vars = len(categories)

# Compute angle for each axis
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]  # Complete the loop

# Prepare the data for plotting
normalized_values_1 += normalized_values_1[:1]
normalized_values_2 += normalized_values_2[:1]

# Initialize the radar chart
fig, ax = plt.subplots(figsize=(12, 9), subplot_kw=dict(polar=True))
ax.grid(True, linewidth=4) 
# Draw one axe per variable and add labels
# ax.tick_params(pad=15)
# plt.xticks(angles[:-1], categories)

# Draw y-labels
ax.set_rlabel_position(30)
plt.ylim(0, 1)
ax.set_yticklabels([""] * len(ax.get_yticks()))

# Plot data
ax.plot(angles, normalized_values_1, 'r', linewidth=2, linestyle='solid', label='SOTA')
ax.plot(angles, normalized_values_2, 'g', linewidth=2, linestyle='solid', label='Qwen-3D')

# Fill the area
ax.fill(angles, normalized_values_1, 'b', alpha=0.2)
ax.fill(angles, normalized_values_2, 'b', alpha=0.2)

ax.set_xticks(angles[:-1])
ax.set_xticklabels([""] * len(categories), fontsize=12)

custom_pads = [1.2, 1.08, 1.05, 1.1, 1.07, 1.05]  # Adjust these values as needed

# Manually set the labels with custom padding
for i, (angle, category, pad) in enumerate(zip(angles[:-1], categories, custom_pads)):
    # Use ax.text() to place the labels at a custom distance
    ax.text(
        angle,  # Angle of the label
        1.05 * pad,  # Radius for positioning the label (customized by pad)
        category,  # Text of the label
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=15
    )
values_2 = np.array([24.7, 63.4, 30.7, 89.8, 59.8])
for i, (angle, value) in enumerate(zip(angles, values_1)):
    ax.scatter(angle, normalized_values_1[i], color='r', s=25, zorder=5)
    ax.text(angle, normalized_values_1[i] - 0.1, f'{value}%', color='black', size=12, 
            horizontalalignment='center', verticalalignment='center')
for i, (angle, value) in enumerate(zip(angles, values_2)):
    ax.scatter(angle, normalized_values_2[i], color='g', s=25, zorder=5)
    ax.text(angle, normalized_values_2[i] + 0.1, f'{value}%', color='black', size=12, 
            horizontalalignment='center', verticalalignment='center')

# Show the plot
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=24)

plt.savefig("radar_results.png", format="png", dpi=300, bbox_inches="tight")