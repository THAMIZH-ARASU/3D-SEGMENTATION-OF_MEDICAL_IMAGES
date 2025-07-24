import torch

def get_optimizer(config, model):
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