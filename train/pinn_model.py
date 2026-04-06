"""
PINN neural network model implementation.
Contains the FeedForward neural network class for Physics-Informed Neural Networks.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.markers import MarkerStyle
from config import DEVICE, MODEL_CONFIG


class FeedForward(nn.Module):
    """Physics-Informed Neural Network for solving PDEs with adaptive mesh refinement."""

    def __init__(self, mesh_x, mesh_y, problem):
        """Initialize the PINN model.

        Args:
            mesh_x: x-coordinates of mesh points
            mesh_y: y-coordinates of mesh points
            problem: PDEProblem driving residual and boundary losses
        """
        super(FeedForward, self).__init__()

        # History tracking
        self.total_error_history = []
        self.relative_l2_error_history = []
        self.total_error_rms_history = []
        self.relative_error_rms_history = []
        self.boundary_error_history = []
        self.train_loss_history = []
        self.total_residual_history = []
        self.boundary_residual_history = []
        self.fixed_total_residual_history = []
        self.relative_fixed_l2_residual_history = []
        self.fixed_boundary_residual_history = []
        self.fixed_rms_residual_history = []
        self.relative_fixed_rms_residual_history = []
        self.mesh_point_history = []
        self.mesh_point_count_history = []
        self.iteration_point_count_history = []
        self.iteration_runtime_history = []
        self.cumulative_runtime_history = []

        # Training components
        self.optimizer = None
        self.fes = None  # Will be set during training
        self.problem = problem

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

        # Store mesh coordinates as buffers so model.to(DEVICE) moves them too.
        self.register_buffer("mesh_x", torch.empty(0, dtype=torch.float32))
        self.register_buffer("mesh_y", torch.empty(0, dtype=torch.float32))
        self.set_mesh_points(mesh_x, mesh_y)

        # Cache the training dataset on the active device once per model.
        self._cached_dataset_id = None
        self._cached_dataset_xy = None
        self._cached_dataset_u = None

    def set_mesh_points(self, mesh_x, mesh_y):
        """Update collocation points on the configured runtime device."""
        self.mesh_x = torch.as_tensor(mesh_x, dtype=torch.float32, device=DEVICE)
        self.mesh_y = torch.as_tensor(mesh_y, dtype=torch.float32, device=DEVICE)

    def _get_cached_dataset_tensors(self, dataset):
        """Move the static supervised dataset to the active device once."""
        if self._cached_dataset_id != id(dataset):
            if hasattr(dataset, "tensors") and len(dataset.tensors) == 2:
                xy, u = dataset.tensors
            else:
                xy, u = dataset[:]

            self._cached_dataset_xy = torch.as_tensor(
                xy, dtype=torch.float32, device=DEVICE
            )
            self._cached_dataset_u = torch.as_tensor(
                u, dtype=torch.float32, device=DEVICE
            )
            self._cached_dataset_id = id(dataset)

        return self._cached_dataset_xy, self._cached_dataset_u

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
        if self.problem is None:
            raise RuntimeError("FeedForward model requires an attached PDE problem")

        if use_meshgrid:
            x, y = torch.meshgrid(x, y, indexing="ij")
            x = x.flatten()
            y = y.flatten()

        return self.problem.pde_residual(self, x, y)

    def loss_data(self, dataset):
        """Compute data loss using the provided dataset.

        Args:
            dataset: PyTorch dataset with mesh coordinates and FEM solutions

        Returns:
            Data loss value
        """
        xy_all, u_all = self._get_cached_dataset_tensors(dataset)
        torch.manual_seed(42)
        idx = torch.randint(len(xy_all), (self.num_data,), device=xy_all.device)
        xy = xy_all[idx]
        u = u_all[idx]

        x, y = xy.unbind(dim=1)

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
        if self.problem is None:
            raise RuntimeError("FeedForward model requires an attached PDE problem")
        return self.problem.boundary_loss(self, self.num_bd)

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
