import numpy as np
import plotly.graph_objects as go

# Define two sets of 3D points
set1 = np.array([[0, 0, 0], [1, 2, 3], [2, 1, 0], [3, 3, 3]])
set2 = np.array([[1, 0, 2], [2, 3, 4], [3, 2, 1], [4, 4, 4]])

# Ensure the sets have the same number of points
if set1.shape != set2.shape:
    raise ValueError("Both sets of points must have the same shape.")

# Initialize an empty Plotly figure
fig = go.Figure()

# Add points from set1 to the plot
fig.add_trace(go.Scatter3d(
    x=set1[:, 0], y=set1[:, 1], z=set1[:, 2],
    mode='markers+text',
    marker=dict(size=5, color='blue'),
    text=[f'Set1-{i}' for i in range(len(set1))],
    textposition="top center",
    name='Set1'
))

# Add points from set2 to the plot
fig.add_trace(go.Scatter3d(
    x=set2[:, 0], y=set2[:, 1], z=set2[:, 2],
    mode='markers+text',
    marker=dict(size=5, color='red'),
    text=[f'Set2-{i}' for i in range(len(set2))],
    textposition="top center",
    name='Set2'
))

# Add line segments connecting corresponding points
for p1, p2 in zip(set1, set2):
    fig.add_trace(go.Scatter3d(
        x=[p1[0], p2[0]],
        y=[p1[1], p2[1]],
        z=[p1[2], p2[2]],
        mode='lines',
        line=dict(color='green', width=2),
        showlegend=False
    ))

# Update layout for better visualization
fig.update_layout(
    title='3D Point Connections',
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

# Show the plot
fig.show()
