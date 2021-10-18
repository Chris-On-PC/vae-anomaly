import yaml
import argparse
import numpy as np

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms
from utils.loaders import XrayImageDataset, CustomLoader, Img_To_1

def main(args):
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name'])


    # For reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    model = vae_models[config['model_params']['name']](**config['model_params'])
    experiment = VAEXperiment(model, config['exp_params'])

    runner = Trainer(default_root_dir =f"{tb_logger.save_dir}",
                    gpus=1 if torch.cuda.is_available() else 0,
                    logger=tb_logger,
                    log_every_n_steps = 100,
                    num_sanity_val_steps=5,
                    **config['trainer_params'])

    transform = transforms.Compose([transforms.ToTensor(), Img_To_1()])

    dataset = XrayImageDataset(img_dir = r"D:\Documents\Anomaly\data",
                             transform=transform,
                             patch_size=config['exp_params']['patch_size'],
                             overlap=config['exp_params']['overlap'])

    dataset_size = len(dataset)
    split = int(np.floor(config['exp_params']['val_split'] * dataset_size))
    train_size = dataset_size - split
    test_size = split

    training_set, validation_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = CustomLoader(training_set,
                          batch_size= config['exp_params']['batch_size'])



    val_loader =  CustomLoader(validation_set, batch_size= config['exp_params']['batch_size'],
                                                 drop_last=True)


    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, train_loader, val_loader)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/vae.yaml')

    args = parser.parse_args()

    main(args)