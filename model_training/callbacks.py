"""
Utility module for creating PyTorch Lightning callbacks used in the training pipeline. Provides a centralized function to configure model checkpointing, early stopping, and learning rate monitoring callbacks based on training configuration.

This module is used by the SegmentationTrainer to set up training callbacks that handle:
- Model checkpointing based on monitored metrics
- Early stopping to prevent overfitting
- Learning rate monitoring for debugging and visualization
"""

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

def get_callbacks(config):
    """
    Create a list of PyTorch Lightning callbacks based on the training configuration.

    This function creates three essential callbacks for training:
    1. ModelCheckpoint: Saves the best model based on monitored metric
    2. EarlyStopping: Stops training when metric doesn't improve
    3. LearningRateMonitor: Logs learning rate changes

    Args:
        config: Training configuration object containing callback parameters.
            Must have attributes: checkpoint_dir, monitor_metric, monitor_mode,
            early_stopping_patience.

    Returns:
        list: List of configured PyTorch Lightning callbacks.

    Example:
        >>> callbacks = get_callbacks(config)
        >>> trainer = pl.Trainer(callbacks=callbacks)
    """
    callbacks = []
    callbacks.append(ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        save_top_k=1,
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        filename="{epoch}-{val_loss:.4f}",
        save_last=True
    ))
    callbacks.append(EarlyStopping(
        monitor=config.monitor_metric,
        patience=config.early_stopping_patience,
        mode=config.monitor_mode,
        verbose=True
    ))
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    return callbacks 