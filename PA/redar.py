import matplotlib.pyplot as plt
import numpy as np

"""
Radar chart for model performance comparison.
This code creates a radar chart to visualize the performance of three models: LLM-Visual-RFT, Visual-RFT, and ContextDET.
"""
# define the data
labels = ['Precision', 'Recall', 'F1', 'AMA']
llm_data = [0.965, 1, 0.982, 0.98]      # LLM-Visual-RFT
vis_data = [0.95, 0.487, 0.642, 0.9]       # Visual-RFT
ctx_data = [0.795, 0.969, 0.872, 0]       # ContextDET

# calculate the number of variables
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]  

# create the radar chart
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)

# draw one axe per variable and add labels
def plot_radar(data, color, label):
    data += data[:1]  
    ax.plot(angles, data, color=color, linewidth=2, label=label)
    ax.fill(angles, data, color=color, alpha=0.1)

plot_radar(llm_data, 'tab:blue', 'LLM-Visual-RFT')
plot_radar(vis_data, 'tab:orange', 'Visual-RFT')
plot_radar(ctx_data, 'tab:red', 'ContextDET')

# set the labels and title
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)
plt.xticks(angles[:-1], labels)
plt.ylim(0, 1.1)

# add a legend and title
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.title('Model Performance Radar Chart', pad=20)

plt.show()