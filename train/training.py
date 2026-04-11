"""
Training functions for PINN models.
Contains functions for training neural networks and managing optimizers.
"""

import copy

import torch
from config import TRAINING_CONFIG, VALIDATION_CONFIG


def evaluate_validation_score(
    model,
    *,
    validation_dataset=None,
    validation_residual_points=None,
):
    """Evaluate a held-out validation score without using oracle reference data."""
    if validation_dataset is None and validation_residual_points is None:
        return None

    model.eval()
    with torch.enable_grad():
        if validation_dataset is not None:
            data_loss = model.loss_data_on_dataset(validation_dataset)
        else:
            data_loss = torch.tensor(0.0, device=next(model.parameters()).device)

        if validation_residual_points is not None:
            val_x, val_y = validation_residual_points
            residual_loss = model.loss_interior_on_points(val_x, val_y)
        else:
            residual_loss = torch.tensor(0.0, device=next(model.parameters()).device)

        validation_score = model.w_data * data_loss + model.w_interior * residual_loss

    model.train()

    return {
        "validation_score": float(validation_score.detach().cpu().item()),
        "validation_data_loss": float(data_loss.detach().cpu().item()),
        "validation_residual_loss": float(residual_loss.detach().cpu().item()),
    }


def train_model(
    model,
    dataset,
    epochs=None,
    optimizer=None,
    *,
    validation_dataset=None,
    validation_residual_points=None,
    validation_check_interval=None,
    **kwargs,
):
    """Train the PINN model using the specified optimizer.

    Args:
        model: PINN model to train
        dataset: Training dataset
        epochs: Number of training epochs. If None, uses config default.
        optimizer: Optimizer type ("Adam" or "L-BFGS")
        **kwargs: Additional optimizer parameters

    Returns:
        dict | None: Best validation summary if validation is enabled
    """
    if epochs is None:
        epochs = TRAINING_CONFIG["epochs"]

    if optimizer is None:
        optimizer = TRAINING_CONFIG["optimizer"]
    if validation_check_interval is None:
        validation_check_interval = VALIDATION_CONFIG["check_interval"]

    # Set default learning rate if not provided
    if "lr" not in kwargs:
        kwargs["lr"] = TRAINING_CONFIG["lr"]

    optimizer_name = optimizer.replace("-", "").upper()

    # Initialize optimizer
    if optimizer_name == "ADAM":
        model.optimizer = torch.optim.Adam(model.parameters(), **kwargs)
    elif optimizer_name == "LBFGS":
        model.optimizer = torch.optim.LBFGS(model.parameters(), **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    num_steps = max(int(epochs), 0)
    print(f"Training with {optimizer} optimizer for {num_steps} epochs...")

    if num_steps == 0:
        return None

    best_validation = None
    best_state_dict = None
    validation_enabled = (
        validation_dataset is not None or validation_residual_points is not None
    )

    # Training loop: run exactly the configured number of optimizer steps.
    for epoch in range(1, num_steps + 1):
        # Use closure function for optimization step
        model.optimizer.step(lambda: model.closure(dataset))

        # Track progress and accumulate loss data for plotting
        if epoch == 1 or epoch % 1000 == 0 or epoch == num_steps:
            loss_interior, loss_data, loss_bc = model.compute_losses(dataset)
            total_loss = (
                model.w_data * loss_data
                + model.w_interior * loss_interior
                + model.w_bc * loss_bc
            )

            # Store training history
            model.train_loss_history.append(
                [
                    total_loss.cpu().detach().numpy(),
                    loss_interior.cpu().detach().numpy(),
                    loss_data.cpu().detach().numpy(),
                    loss_bc.cpu().detach().numpy(),
                ]
            )

            print(
                f"Epoch {epoch:5d}: Total Loss = {total_loss.detach().cpu().numpy():.6e}"
            )

        if validation_enabled and (
            epoch == 1
            or epoch % max(int(validation_check_interval), 1) == 0
            or epoch == num_steps
        ):
            validation_result = evaluate_validation_score(
                model,
                validation_dataset=validation_dataset,
                validation_residual_points=validation_residual_points,
            )
            if validation_result is not None:
                validation_result["epoch"] = epoch
                if (
                    best_validation is None
                    or validation_result["validation_score"]
                    < best_validation["validation_score"]
                ):
                    best_validation = validation_result
                    best_state_dict = copy.deepcopy(model.state_dict())
                print(
                    "            Validation Score = "
                    f"{validation_result['validation_score']:.6e} "
                    f"(data={validation_result['validation_data_loss']:.6e}, "
                    f"residual={validation_result['validation_residual_loss']:.6e})"
                )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(
            "Restored best validation checkpoint: "
            f"epoch={best_validation['epoch']}, "
            f"score={best_validation['validation_score']:.6e}"
        )

    return best_validation


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

    optimizer_name = optimizer_type.replace("-", "").upper()

    if optimizer_name == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), **kwargs)
    elif optimizer_name == "LBFGS":
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
