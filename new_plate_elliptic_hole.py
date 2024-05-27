import torch
import numpy as np
from scipy.stats import qmc
from plotly.express.colors import sequential
import plotly.graph_objects as go
import plotly.figure_factory as ff
from tqdm import tqdm
import matplotlib.pyplot as plt

from global_constants import L, R, N1, N2
DTYPE = torch.float32

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
            np.loadtxt(f"data/inputs_Rx={self.R_x:.2f}.csv", delimiter=","), dtype=DTYPE
        )
        data_output = torch.as_tensor(
            np.loadtxt(f"data/outputs_Rx={self.R_x:.2f}.csv", delimiter=","), dtype=DTYPE
        )

        data_hole = torch.as_tensor(
            np.loadtxt(f"data/hole_Rx={self.R_x:.2f}.csv", delimiter=","), dtype=DTYPE
        )
        return data_input, data_output, data_hole
    
    
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
        r_collo = self.R_x * np.ones((indices.size, 1)) 
        collocation = torch.tensor(np.hstack((points, r_collo)),dtype=DTYPE)

        # Boundary points
        x_top = L * torch.tensor(qmc.LatinHypercube(d=1).random(self.N), dtype=DTYPE)
        y_top = self.L * torch.ones((self.N, 1))
        r_top = self. R_x * torch.ones((self.N, 1))
        top = torch.column_stack([x_top, y_top, r_top])

        x_right = self.L * torch.ones((self.N, 1))
        y_right = L * torch.tensor(qmc.LatinHypercube(d=1).random(self.N), dtype=DTYPE)
        r_right = self. R_x * torch.ones((self.N, 1))
        right = torch.column_stack([x_right, y_right, r_right])

        NN = int(self.N * (self.L - self.R_y) / self.L)
        left_rand_samp = qmc.LatinHypercube(d=1).random(NN)
        x_left = torch.zeros((NN, 1))
        y_left = self.R_y + (L - self.R_y) * torch.tensor(left_rand_samp, dtype=DTYPE)
        r_left = self. R_x * torch.ones((NN, 1))
        left = torch.column_stack([x_left, y_left, r_left])

        NN = int(self.N * (self.L - self.R_x) / self.L)
        bottom_rand_samp = qmc.LatinHypercube(d=1).random(NN)
        x_bottom = self.R_x + (L - self.R_x) * torch.tensor(bottom_rand_samp, dtype=DTYPE)
        y_bottom = torch.zeros((NN, 1))
        r_bottom = self. R_x * torch.ones((NN, 1))
        bottom = torch.column_stack([x_bottom, y_bottom, r_bottom])


        hole_rand_samp = qmc.LatinHypercube(d=1).random(int(self.N * np.pi * self.R_x / L)).ravel()
        phi = 0.5 * np.pi * torch.tensor(hole_rand_samp, dtype=DTYPE)
        #phi = np.linspace(0, 0.5 * np.pi, int(self.N*0.5))
        x_hole = self.R_x * torch.cos(phi)
        y_hole = self.R_y * torch.sin(phi)
        n_hole = torch.stack([-self.R_y * torch.cos(phi), -self.R_x * torch.sin(phi)]).T
        n_hole = n_hole / torch.linalg.norm(n_hole, axis=1)[:, None]
        r_hole = self. R_x * torch.ones((phi.size(dim=0), 1))
        hole = torch.column_stack([x_hole, y_hole, r_hole, n_hole])

        
        return collocation, top, right, left, bottom, hole


    def plot_plate_with_hole(self, collocation, top, right, left, bottom, hole):
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


    

