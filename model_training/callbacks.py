from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

def get_callbacks(config):
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