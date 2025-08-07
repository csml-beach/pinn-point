"""
Training functions for PINN models.
Contains functions for training neural networks and managing optimizers.
"""

import torch
from config import DEVICE, TRAINING_CONFIG


def train_model(model, dataset, epochs=None, optimizer="Adam", **kwargs):
    """Train the PINN model using the specified optimizer.
    
    Args:
        model: PINN model to train
        dataset: Training dataset
        epochs: Number of training epochs. If None, uses config default.
        optimizer: Optimizer type ("Adam" or "L-BFGS")
        **kwargs: Additional optimizer parameters
        
    Returns:
        None (modifies model in-place)
    """
    if epochs is None:
        epochs = TRAINING_CONFIG["epochs"]
    
    # Set default learning rate if not provided
    if "lr" not in kwargs:
        kwargs["lr"] = TRAINING_CONFIG["lr"]
    
    # Initialize optimizer
    if optimizer == "Adam":
        model.optimizer = torch.optim.Adam(model.parameters(), **kwargs)
    elif optimizer == "L-BFGS":
        model.optimizer = torch.optim.LBFGS(model.parameters(), **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    print(f"Training with {optimizer} optimizer for {epochs} epochs...")
    
    # Training loop
    for epoch in range(epochs + 1):
        # Use closure function for optimization step
        model.optimizer.step(lambda: model.closure(dataset))

        # Track progress and accumulate loss data for plotting
        if epoch % 1000 == 0:
            loss_interior, loss_data, loss_bc = model.compute_losses(dataset)
            total_loss = (
                model.w_data * loss_data
                + model.w_interior * loss_interior
                + model.w_bc * loss_bc
            )

            # Store training history
            model.train_loss_history.append([
                total_loss.cpu().detach().numpy(),
                loss_interior.cpu().detach().numpy(),
                loss_data.cpu().detach().numpy(),
                loss_bc.cpu().detach().numpy(),
            ])

            print(f"Epoch {epoch:5d}: Total Loss = {total_loss.detach().cpu().numpy():.6e}")


def setup_optimizer(model, optimizer_type="Adam", **kwargs):
    """Setup optimizer for the model.
    
    Args:
        model: PINN model
        optimizer_type: Type of optimizer ("Adam" or "L-BFGS")
        **kwargs: Optimizer parameters
        
    Returns:
        Configured optimizer
    """
    if "lr" not in kwargs:
        kwargs["lr"] = TRAINING_CONFIG["lr"]
    
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), **kwargs)
    elif optimizer_type == "L-BFGS":
        optimizer = torch.optim.LBFGS(model.parameters(), **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    return optimizer


def evaluate_model(model, dataset):
    """Evaluate the model on the given dataset.
    
    Args:
        model: PINN model
        dataset: Evaluation dataset
        
    Returns:
        dict: Dictionary containing loss components
    """
    model.eval()
    with torch.no_grad():
        loss_interior, loss_data, loss_bc = model.compute_losses(dataset)
        total_loss = (
            model.w_data * loss_data
            + model.w_interior * loss_interior
            + model.w_bc * loss_bc
        )
    
    model.train()
    
    return {
        "total_loss": total_loss.item(),
        "interior_loss": loss_interior.item(),
        "data_loss": loss_data.item(),
        "boundary_loss": loss_bc.item(),
    }
