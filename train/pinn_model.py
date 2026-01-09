"""
PINN neural network model implementation.
Contains the FeedForward neural network class for Physics-Informed Neural Networks.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from config import DEVICE, MODEL_CONFIG


class FeedForward(nn.Module):
    """Physics-Informed Neural Network for solving PDEs with adaptive mesh refinement."""

    def __init__(self, mesh_x, mesh_y):
        """Initialize the PINN model.

        Args:
            mesh_x: x-coordinates of mesh points
            mesh_y: y-coordinates of mesh points
        """
        super(FeedForward, self).__init__()

        # History tracking
        self.total_error_history = []
        self.boundary_error_history = []
        self.train_loss_history = []
        self.total_residual_history = []
        self.boundary_residual_history = []
        self.mesh_point_history = []
        self.mesh_point_count_history = []

        # Training components
        self.optimizer = None
        self.fes = None  # Will be set during training

        # Loss function weights
        self.w_data = MODEL_CONFIG["w_data"]
        self.w_interior = MODEL_CONFIG["w_interior"]
        self.w_bc = MODEL_CONFIG["w_bc"]

        # Network architecture parameters
        self.hidden_size = MODEL_CONFIG["hidden_size"]
        self.num_data = MODEL_CONFIG["num_data"]
        self.num_bd = MODEL_CONFIG["num_bd"]

        # Neural network layers (2 input features -> hidden -> hidden -> 1 output)
        self.b1 = nn.Linear(2, self.hidden_size)
        self.b2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.b3 = nn.Linear(self.hidden_size, 1)

        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.b1.weight)
        nn.init.xavier_uniform_(self.b2.weight)
        nn.init.xavier_uniform_(self.b3.weight)

        # Store mesh coordinates
        self.mesh_x = mesh_x
        self.mesh_y = mesh_y

    def forward(self, x, y):
        """Forward pass through the neural network.

        Args:
            x: x-coordinates
            y: y-coordinates

        Returns:
            Neural network output u(x,y)
        """
        xy = torch.stack((x, y), dim=1)
        h1 = torch.tanh(self.b1(xy))
        h2 = torch.tanh(self.b2(h1))
        u = self.b3(h2)
        return u

    def compute_derivative(self, u, x, n):
        """Compute the n-th order derivative of u with respect to x using automatic differentiation.

        Args:
            u: Function values
            x: Input variable
            n: Order of derivative

        Returns:
            n-th order derivative
        """
        if n == 0:
            return u
        else:
            du_dx = torch.autograd.grad(
                u,
                x,
                torch.ones_like(u).to(DEVICE),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            return self.compute_derivative(du_dx, x, n - 1)

    def PDE_residual(self, x, y, use_meshgrid=False):
        """Compute the PDE residual for the Poisson equation: ∇²u + f = 0.

        NOTE: This is the default Poisson problem implementation.
        For new PDEs, use the extensible problems module:
            from problems import get_problem
            problem = get_problem("poisson")  # or your custom problem
            residual = problem.pde_residual(model, x, y)

        Args:
            x: x-coordinates
            y: y-coordinates
            use_meshgrid: Whether to use meshgrid for coordinates

        Returns:
            PDE residual values
        """
        if use_meshgrid:
            X, Y = torch.meshgrid(x, y)
            xy = torch.stack((X.flatten(), Y.flatten()), dim=1)

        x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
        y = torch.tensor(y, dtype=torch.float32).to(DEVICE)
        x.requires_grad = True
        y.requires_grad = True

        u = self.forward(x, y)

        # Compute second derivatives
        d2u_dx2 = self.compute_derivative(u, x, 2)
        d2u_dy2 = self.compute_derivative(u, y, 2)

        # PDE: ∇²u + f = 0, where f = x*y
        residual = d2u_dx2 + d2u_dy2 + x * y
        return residual

    def loss_data(self, dataset):
        """Compute data loss using the provided dataset.

        Args:
            dataset: PyTorch dataset with mesh coordinates and FEM solutions

        Returns:
            Data loss value
        """
        torch.manual_seed(42)
        idx = torch.randint(len(dataset), (self.num_data,))
        xy, u = dataset[idx]

        x, y = xy.unbind(axis=1)
        x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
        y = torch.tensor(y, dtype=torch.float32).to(DEVICE)
        u = torch.tensor(u, dtype=torch.float32).to(DEVICE)

        u_pred = self.forward(x, y)
        loss_data = torch.mean(torch.square(u - u_pred))
        return loss_data

    def loss_interior(self):
        """Compute interior loss based on PDE residual.

        Returns:
            Interior loss value
        """
        res = self.PDE_residual(self.mesh_x, self.mesh_y)
        loss_residual = torch.mean(torch.square(res))
        return loss_residual

    def loss_boundary_condition(self):
        """Compute boundary condition loss (Dirichlet BC: u = 0 on bottom boundary).

        NOTE: This is the default Poisson problem BC. For new PDEs, use:
            from problems import get_problem
            problem = get_problem("your_problem")
            bc_loss = problem.boundary_loss(model, num_points)

        Returns:
            Boundary condition loss value
        """
        self.x_bottom = torch.linspace(0, 5, self.num_bd).reshape(-1)
        self.y_bottom = torch.zeros(1, self.num_bd).reshape(-1)

        bc_pred_bottom = self.forward(
            self.x_bottom.to(DEVICE), self.y_bottom.to(DEVICE)
        )
        loss_bc_bottom = torch.mean(torch.square(bc_pred_bottom))
        return loss_bc_bottom

    def compute_losses(self, dataset):
        """Compute all loss components.

        Args:
            dataset: Training dataset

        Returns:
            tuple: (loss_interior, loss_data, loss_bc)
        """
        loss_interior = self.loss_interior()
        loss_data = self.loss_data(dataset)
        loss_bc = self.loss_boundary_condition()
        return loss_interior, loss_data, loss_bc

    def closure(self, dataset):
        """Closure function for optimizer.

        Args:
            dataset: Training dataset

        Returns:
            Total loss value
        """
        self.optimizer.zero_grad()
        loss_interior, loss_data, loss_bc = self.compute_losses(dataset)
        total_loss = (
            self.w_data * loss_data
            + self.w_interior * loss_interior
            + self.w_bc * loss_bc
        )
        total_loss.backward(retain_graph=True)
        return total_loss

    def get_training_history(self):
        """Get training loss history.

        Returns:
            tuple: (total_loss, loss_bc, loss_interior, loss_data) arrays
        """
        loss_hist = np.array(self.train_loss_history)
        if len(loss_hist) == 0:
            return [], [], [], []
        total_loss, loss_interior, loss_data, loss_bc = np.split(loss_hist, 4, axis=1)
        return total_loss, loss_bc, loss_interior, loss_data

    def plot_pinn_losses(self):
        """Plot PINN training losses."""
        total_loss, loss_bc, loss_interior, loss_data = self.get_training_history()

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(
            total_loss,
            marker=MarkerStyle("o", fillstyle="none"),
            color="black",
            label="Total Loss",
        )
        plt.yscale("log")
        plt.legend()
        plt.title("Total Loss")

        plt.subplot(2, 2, 2)
        plt.plot(
            loss_bc,
            marker=MarkerStyle("o", fillstyle="none"),
            color="red",
            label="Boundary Condition Loss",
        )
        plt.yscale("log")
        plt.legend()
        plt.title("Boundary Condition Loss")

        plt.subplot(2, 2, 3)
        plt.plot(
            loss_interior,
            marker=MarkerStyle("o", fillstyle="none"),
            color="blue",
            label="Interior Loss",
        )
        plt.yscale("log")
        plt.legend()
        plt.title("Interior Loss")

        plt.subplot(2, 2, 4)
        plt.plot(
            loss_data,
            marker=MarkerStyle("o", fillstyle="none"),
            color="green",
            label="Data Loss",
        )
        plt.yscale("log")
        plt.legend()
        plt.title("Data Loss")

        plt.tight_layout()
        plt.show()

    def plot_pde_residuals(self):
        """Plot PDE residuals and mesh information."""
        plt.figure(figsize=(15, 4))

        plt.subplot(1, 3, 1)
        plt.plot(
            self.mesh_point_count_history,
            marker=MarkerStyle("o", fillstyle="none"),
            color="black",
            label="mesh point count",
        )
        plt.legend()
        plt.title("Mesh Point Count")

        plt.subplot(1, 3, 2)
        plt.plot(
            self.total_residual_history,
            marker=MarkerStyle("o", fillstyle="none"),
            color="red",
            label="total residual",
        )
        plt.yscale("log")
        plt.legend()
        plt.title("Total Residual")

        plt.subplot(1, 3, 3)
        plt.plot(
            self.boundary_residual_history,
            marker=MarkerStyle("o", fillstyle="none"),
            color="red",
            label="boundary residual",
        )
        plt.yscale("log")
        plt.legend()
        plt.title("Boundary Residual")

        plt.tight_layout()
        plt.show()

    def plot_model_error(self):
        """Plot model error compared to FEM solution."""
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(
            self.total_error_history,
            marker=MarkerStyle("o", fillstyle="none"),
            color="black",
            label="total error",
        )
        plt.yscale("log")
        plt.legend()
        plt.title("Total Error")

        plt.subplot(1, 2, 2)
        plt.plot(
            self.boundary_error_history,
            marker=MarkerStyle("o", fillstyle="none"),
            color="red",
            label="boundary error",
        )
        plt.yscale("log")
        plt.legend()
        plt.title("Boundary Error")

        plt.tight_layout()
        plt.show()
