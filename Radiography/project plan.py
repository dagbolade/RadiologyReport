import pandas as pd
import matplotlib.pyplot as plt

# Updated project plan data with corrected durations and timeline
data = {
    'Task': ['Research', 'Data Collection', 'Data Preprocessing', 'Model Development',
             'Training and Validation', 'Evaluation and Testing', 'Deployment', 'Report Writing', 'Submission'],
    'Start': pd.to_datetime(['2024-05-21', '2024-06-04', '2024-06-18', '2024-07-02',
                             '2024-07-10', '2024-07-20', '2024-08-17', '2024-07-28', '2024-08-31']),
    'End': pd.to_datetime(['2024-06-03', '2024-06-17', '2024-07-01', '2024-07-30',
                           '2024-08-01', '2024-08-03', '2024-08-31', '2024-08-30', '2024-08-31'])
}

df = pd.DataFrame(data)
df['Duration'] = (df['End'] - df['Start']).dt.days + 1  # Add 1 to include both start and end days

# Define colors for each task
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

# Plotting the Gantt chart
fig, ax = plt.subplots(figsize=(14, 8))
for i, task in enumerate(df['Task']):
    ax.barh(task, df['Duration'][i], left=df['Start'][i], color=colors[i])
    # Add text for duration
    ax.text(df['End'][i], i, f' {df["Duration"][i]} days', va='center', fontsize=10, color='black')

ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Task', fontsize=14)
ax.set_title('Project Plan Gantt Chart', fontsize=16)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the Gantt chart as an image
plt.savefig('final_gantt_chart_with_duration.png', dpi=300, bbox_inches='tight')
plt.show()