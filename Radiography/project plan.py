import pandas as pd
import matplotlib.pyplot as plt

# Create a Gantt chart for the project plan with durations
data = {
    'Task': ['Research', 'Data Collection', 'Data Preprocessing', 'Model Development', 'Training and Validation', 'Evaluation and Testing', 'Report Writing', 'Submission'],
    'Start': pd.to_datetime(['2024-05-21', '2024-06-01', '2024-06-08', '2024-06-15', '2024-06-25', '2024-07-05', '2024-07-12', '2024-07-18']),
    'End': pd.to_datetime(['2024-05-31', '2024-06-07', '2024-06-14', '2024-06-24', '2024-07-04', '2024-07-11', '2024-07-17', '2024-07-20'])
}
df = pd.DataFrame(data)
df['Duration'] = (df['End'] - df['Start']).dt.days

# Define colors for each task
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# Plotting the Gantt chart
fig, ax = plt.subplots(figsize=(14, 8))
for i, task in enumerate(df['Task']):
    ax.barh(task, df['Duration'][i], left=df['Start'][i], color=colors[i])
    # Add text for duration
    ax.text(df['End'][i], i, f' {df["Duration"][i]} days', va='center', fontsize=12, color='black')

ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Task', fontsize=14)
ax.set_title('Project Plan Gantt Chart', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the Gantt chart as an image
plt.savefig('enhanced_gantt_chart_with_duration.png')
plt.show()
