"""
optimizers.py

Optimizer and scheduler configuration for medical image segmentation training. Provides a centralized function to create optimizers and learning rate schedulers based on training configuration.

This module supports multiple optimizer types (Adam, AdamW, SGD) and scheduler types
(StepLR, ReduceLROnPlateau) commonly used in deep learning training pipelines.
"""

import torch

def get_optimizer(config, model):
    """
    Create optimizer and scheduler based on training configuration.

    This function creates an optimizer and optional learning rate scheduler
    based on the provided configuration. It supports multiple optimizer types
    and scheduler configurations commonly used in deep learning training.

    Args:
        config: Training configuration object containing optimizer parameters.
            Must have attributes: optimizer, learning_rate, weight_decay,
            scheduler, scheduler_params.
        model: PyTorch model whose parameters will be optimized.

    Returns:
        tuple: (optimizer, scheduler) where scheduler may be None.

    Supported Optimizers:
        - 'adam': Adam optimizer with adaptive learning rates
        - 'adamw': AdamW optimizer with decoupled weight decay
        - 'sgd': Stochastic Gradient Descent with momentum

    Supported Schedulers:
        - 'step': StepLR - reduces learning rate by gamma every step_size epochs
        - 'plateau': ReduceLROnPlateau - reduces learning rate when metric plateaus

    Example:
        >>> optimizer, scheduler = get_optimizer(config, model)
        >>> if scheduler:
        ...     return {"optimizer": optimizer, "lr_scheduler": scheduler}
        ... else:
        ...     return optimizer
    """
    if config.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    scheduler = None
    if config.scheduler:
        if config.scheduler.lower() == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **config.scheduler_params)
        elif config.scheduler.lower() == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config.scheduler_params)
        # Add more schedulers as needed
    return optimizer, scheduler 