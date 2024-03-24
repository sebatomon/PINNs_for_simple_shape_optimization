import torch
import numpy as np
from scipy.stats import qmc
from plotly.express.colors import sequential
import plotly.graph_objects as go
import plotly.figure_factory as ff
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
from global_constants import L, R

class Plate:
    def __init__(self, R_x, N, M):
        # Elliptical axis in x direction
        self.R_x = R_x
        # Elliptical axis in y direction
        self.R_y = R**2 / R_x
        #length of the plate 
        self.L = L
        # Edge samples
        self.N = int(N)
        # Number of collocation points
        self.M = M
        self.collopoints = None


    def load_reference_data(self):
        # Load reference data
        data_input = torch.as_tensor(
            np.loadtxt(f"data/inputs_Rx={self.R_x}.csv", delimiter=","), dtype=torch.float64
        )
        data_output = torch.as_tensor(
            np.loadtxt(f"data/outputs_Rx={self.R_x}.csv", delimiter=","), dtype=torch.float64
        )
        return data_input, data_output
    
    
    def collocation_weights(self, x):
        p = 1 / (x[:, 0] ** 2 / self.R_x**2 + x[:, 1] ** 2 / self.R_y**2)
        mask = (((x[:, 0] ** 2) / (self.R_x**2)) + ((x[:, 1] ** 2) / (self.R_y**2))) < 1
        p[mask] = 0
        return p / np.sum(p)
    
    def generate_dataset(self):

        # Create collocation points
        samples = qmc.LatinHypercube(d=2).random(1000 * self.M)
        indices = np.random.choice(1000 * self.M, self.M, p=self.collocation_weights(samples), replace=False)
        points = samples[indices]
        collocation = torch.tensor(points, requires_grad=True).double()

        # Boundary points
        x_top = torch.linspace(0, self.L, self.N, requires_grad=True).double()
        y_top = self.L * torch.ones((self.N, 1), requires_grad=True).double()
        top = torch.column_stack([x_top, y_top]).double()

        x_right = self.L * torch.ones((self.N, 1)).double()
        y_right = torch.linspace(0, self.L, self.N).double()
        right = torch.column_stack([x_right, y_right]).double()

        NN = int(self.N * (self.L - self.R_y) / self.L)
        x_left = torch.zeros((NN, 1)).double()
        y_left = torch.linspace(self.R_y, self.L, NN).double()
        left = torch.column_stack([x_left, y_left]).double()

        NN = int(self.N * (self.L - self.R_x) / self.L)
        x_bottom = torch.linspace(self.R_x, self.L, NN).double()
        y_bottom = torch.zeros((NN, 1)).double()
        bottom = torch.column_stack([x_bottom, y_bottom]).double()


        phi = np.linspace(0, 0.5 * np.pi, int(self.N*0.5))
        x_hole = torch.tensor(self.R_x * np.cos(phi), requires_grad=True).double()
        y_hole = torch.tensor(self.R_y * np.sin(phi), requires_grad=True).double()
        n_hole = torch.tensor(np.stack([-(self.R_y) * np.cos(phi), -(self.R_x) * np.sin(phi)]).T).double()
        n_hole = n_hole / torch.linalg.norm(n_hole, axis=1)[:, None]
        hole = torch.column_stack([x_hole, y_hole]).double()
        
        return collocation, top, right, left, bottom, hole, n_hole


    def plot_plate_with_hole(self, collocation, top, right, left, bottom, hole, n_hole):
        # Visualize geometry
        with torch.no_grad():
            mode = "markers"
            gray = dict(color="#C9C5BC")
            green = dict(color="#006561")
            black = dict(color="black")
            fig = go.Figure()
            #fig = ff.create_quiver(hole[:, 0], hole[:, 1], n_hole[:, 0], n_hole[:, 1], marker=black)
            fig.add_trace(go.Scatter(x=collocation[:, 0], y=collocation[:, 1], mode=mode, marker=gray))
            fig.add_trace(go.Scatter(x=top[:, 0], y=top[:, 1], mode=mode, marker=green))
            fig.add_trace(go.Scatter(x=bottom[:, 0], y=bottom[:, 1], mode=mode, marker=green))
            fig.add_trace(go.Scatter(x=left[:, 0], y=left[:, 1], mode=mode, marker=green))
            fig.add_trace(go.Scatter(x=right[:, 0], y=right[:, 1], mode=mode, marker=green))
            fig.add_trace(go.Scatter(x=hole[:, 0], y=hole[:, 1], mode=mode, marker=black))
            fig.layout.yaxis.scaleanchor = "x"
            fig.update_layout(
                template="none",
                width=400,
                height=400,
                margin=dict(l=0, r=0, b=0, t=0),
                showlegend=False,
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.show()


    

