import matplotlib.pyplot as plt
from matplotlib import rcParams

# Global font settings
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']  # or 'CMU Serif'
rcParams['font.size'] = 14

# Colors
PRIMARY_COLOR = '#0000ff'  # Blue
PRIMARY_COLOR_LIGHT = '#add8e6'  # Light blue
SECONDARY_COLOR = '#ff7f0e'  # Optional accent


# Axes settings
rcParams['axes.titlesize'] = 16
rcParams['axes.labelsize'] = 14
rcParams['axes.edgecolor'] = 'black'
rcParams['axes.linewidth'] = 1.2

# Tick settings
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.major.size'] = 6
rcParams['ytick.major.size'] = 6

# Legend settings
rcParams['legend.fontsize'] = 12
rcParams['legend.frameon'] = False

# Figure settings
rcParams['figure.figsize'] = (6, 4)
rcParams['figure.dpi'] = 300

# Line settings
rcParams['lines.linewidth'] = 2
rcParams['lines.markersize'] = 6

# Grid settings
rcParams['axes.grid'] = False
rcParams['grid.linestyle'] = '--'
rcParams['grid.alpha'] = 0.7

# Savefig settings
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

# Optional: Set color cycle (if you want consistent multiple line colors)
rcParams['axes.prop_cycle'] = plt.cycler(color=[PRIMARY_COLOR, SECONDARY_COLOR, '#2ca02c', '#d62728'])

