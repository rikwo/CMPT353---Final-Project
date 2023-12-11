import torch
from torch import nn
from torchviz import make_dot
from NHL_model_evaluations import NHLMLP

# python visualization.py

# Create an instance of the model
PATH = "NHL_best_model.pth"
model = NHLMLP(n_inputs=22)
# Load
model.load_state_dict(torch.load(PATH))
model.eval()

# Dummy input for visualization purposes
dummy_input = torch.randn((1, 22))

# Forward pass to generate a graph
output = model(dummy_input)

# Create a graph of the model
graph = make_dot(output, params=dict(model.named_parameters()))

# Save the graph as an image (optional)
graph.render("model_graph", format="png", cleanup=True)

# Display the graph (optional)
graph.view()