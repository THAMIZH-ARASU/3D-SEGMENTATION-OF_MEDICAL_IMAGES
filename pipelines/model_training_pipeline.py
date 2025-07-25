import importlib
from configs.model_training_config import ModelTrainingConfig
from pipelines.data_loading_pipeline import get_dataloader
from model_training.trainer import SegmentationTrainer
from model_training.lightning_module import SegmentationLightningModule
import torchio as tio

# Model registry for extensibility
def get_model_class(model_name: str):
    registry = {
        'dformer3d': 'models.transformers.d_former.network.SegNetwork',
        'res_trans_unet': 'models.transformers.res_trans_unet.network.ResTransUNet',
        # 'unet3d': 'models.cnns.unet_3d.UNet3D',  # Example for future
    }
    if model_name not in registry:
        raise ValueError(f"Model {model_name} not registered.")
    module_path, class_name = registry[model_name].rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def get_largest_divisor(n, max_divisor):
    for d in range(max_divisor, 0, -1):
        if n % d == 0:
            return d
    return 1

def run_training_pipeline(config: ModelTrainingConfig):
    # Set seed for reproducibility
    import pytorch_lightning as pl
    pl.seed_everything(config.seed)

    # Data loaders
    train_loader = get_dataloader(config.data_dir, 'train', config.train_batch_size, config.num_workers, shuffle=True)
    val_loader = get_dataloader(config.data_dir, 'val', config.val_batch_size, config.num_workers, shuffle=False)
    test_loader = get_dataloader(config.data_dir, 'test', config.val_batch_size, config.num_workers, shuffle=False)

    # Automatically set img_size and window_size for models that require it (e.g., res_trans_unet)
    if config.model_name == 'res_trans_unet':
        if 'img_size' not in config.model_params or config.model_params['img_size'] is None:
            for batch in train_loader:
                x = batch['image'][tio.DATA]
                img_size = tuple(x.shape[2:])
                config.model_params['img_size'] = img_size
                break
        # Ensure patch_size is set (default to 8 if not provided)
        if 'patch_size' not in config.model_params or config.model_params['patch_size'] is None:
            config.model_params['patch_size'] = 8
        # Set window_size to a divisor of img_size (use 8 or 16 for 256, etc.)
        if 'window_size' not in config.model_params or config.model_params['window_size'] is None:
            # Use the smallest dimension for safety
            min_dim = min(config.model_params['img_size'])
            # Prefer 8, then 4, then 2, etc.
            for preferred in [8, 16, 4, 2]:
                if min_dim % preferred == 0:
                    config.model_params['window_size'] = preferred
                    break
            else:
                config.model_params['window_size'] = get_largest_divisor(min_dim, min_dim // 2)

    # Instantiate model
    ModelClass = get_model_class(config.model_name)
    base_model = ModelClass(num_classes=config.num_classes, in_chan=config.input_channels, **config.model_params)
    lightning_model = SegmentationLightningModule(base_model, config)

    # Trainer
    trainer = SegmentationTrainer(config, lightning_model, train_loader, val_loader, test_loader)
    trainer.fit()
    trainer.test()
