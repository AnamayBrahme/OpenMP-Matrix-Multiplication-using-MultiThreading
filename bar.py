import matplotlib.pyplot as plt
import numpy as np

# Data from your output
matrix_sizes = ['3x3', '1000x1000']
single_thread_times = [0.000001, 5.85574]
openmp_times = [0.000105, 0.855992]

x = np.arange(len(matrix_sizes))  # label locations
width = 0.35  # bar width

fig, ax = plt.subplots(figsize=(8,5))

# Bars for single-threaded and OpenMP
rects1 = ax.bar(x - width/2, single_thread_times, width, label='Single-threaded', color='tab:blue')
rects2 = ax.bar(x + width/2, openmp_times, width, label='OpenMP (8 threads)', color='tab:orange')

# Add some text for labels, title and axes ticks
ax.set_ylabel('Time (seconds)')
ax.set_title('Matrix Multiplication Execution Time')
ax.set_xticks(x)
ax.set_xticklabels(matrix_sizes)
ax.legend()

# Add value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.6f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0,3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()
