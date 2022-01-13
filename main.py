from torch.utils.data.dataloader import DataLoader
from utils import HfArgumentParser, set_seed
from config import Config
from data import Data, SingleStepDataset
from model.model import *
from train import train
from torch import optim
from evaluate import evaluate

import sys
import os
import torch
import torch.nn as nn
import logging
import optuna

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


# set logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

parser = HfArgumentParser((Config))
config = parser.parse_args_into_dataclasses()[0]
logger = logging.getLogger(__name__)

if not os.path.exists(config.model_dir):
    os.mkdir(config.model_dir)
if not os.path.exists(config.log_dir):
    os.mkdir(config.log_dir)

if os.path.exists(config.filename):
    if not os.path.exists(config.log_dir + '/' + config.dataset):
        os.mkdir(config.log_dir + '/' + config.dataset)
else:
    raise ValueError("Unknown dataset: {}".format(config.filename))

config.log_path = './' + config.log_dir + '/' + config.dataset + '/' + config.model_string + "_" + config.now + ".log"
filehandler = logging.FileHandler(config.log_path, mode="w")
logger.addHandler(filehandler)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        torch.nn.init.constant_(m.bias.data, 0.01)


def define_model(trial):
    config.lr = trial.suggest_loguniform("lr", low=1e-4, high=1e-2)
    config.clip_value = trial.suggest_float("clip_value", low=0.3, high=1.5, step=0.3)
    config.optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
    config.mlp_layer = trial.suggest_int("mlp_layer", low=1, high=3, step=1)
    config.embed_dim = trial.suggest_int("embed_dim", low=16, high=64, step=16)
    config.gamma = trial.suggest_int("gamma", low=0.6, high=1.0, step=0.1)
    config.step_size = trial.suggest_int("step_size", low=10, high=30, step=10)

    if config.dataset == "electricity":
        config.batch_size = 16
    if config.dataset == "traffic":
        config.batch_size = 4
        config.step_size = trial.suggest_int("step_size", low=10, high=20, step=5)
    if config.dataset == "solar_AL":
        config.batch_size = 32
    if config.dataset == "exchange_rate":
        config.batch_size = 32

    return config


def objective(trial):
    config = define_model(trial)
    logger.info(config)
    # Set random seed
    set_seed(config.seed)
    # Set running device
    device = torch.device(config.device)

    # Load data
    train_data, valid_data, test_data, scales, rse_val_d, rae_val_d, rse_test_d, rae_test_d = Data(
        filename=config.filename,
        train_portion=config.train_portion,
        valid_portion=config.valid_portion,
        window=config.window,
        horizon=config.horizon,
        normalize=config.normalize)

    # Convert data to dataloader
    train_dataloader = DataLoader(SingleStepDataset(train_data), batch_size=config.batch_size, shuffle=True)
    valid_dataloader = DataLoader(SingleStepDataset(valid_data), batch_size=config.batch_size, shuffle=False)
    test_dataloader = DataLoader(SingleStepDataset(test_data), batch_size=config.batch_size, shuffle=False)
    data = {
        "train_dataloader": train_dataloader,
        "valid_dataloader": valid_dataloader,
        "test_dataloader": test_dataloader
    }

    # Feature dim
    feature_dim = train_data[1].shape[1]

    # Define model
    if config.model_string == "CGMF":
        model = CGMF(device, config.batch_size, feature_dim, config.window, config.d_inner,
                     config.n_head, config.d_k,
                     config.d_v, config.kernel_sizes, config.channel, config.mlp_layer,
                     config.embed_dim, dropout=config.dropout)
    else:
        raise ValueError("Unknown model_string: {}".format(config.model_string))

    if config.initialize == True:
        model.apply(weights_init)

    model.device = device
    model.to(device)
    scales = torch.tensor(scales, dtype=torch.float).to(device)

    criterionMSE = nn.MSELoss(reduction="sum")
    criterionL1 = nn.L1Loss(reduction="sum")

    # Define optimizer
    weight_p, bias_p = [], []
    for name, p in model.named_parameters():
        if "bias" in name:
            bias_p.append(p)
        else:
            weight_p.append(p)
    optimizer = getattr(optim, config.optimizer_name)(model.parameters(), lr=config.lr)

    # Define Scheduler
    if config.scheduler == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    else:
        scheduler = None

    nParams = sum([p.nelement() for p in model.parameters()])
    logger.info("Number of model parameters is {:d} ".format(nParams))

    # Train
    train(model, data, optimizer, config, scales, rse_val_d, rae_val_d, rse_test_d, rae_test_d, criterionMSE,
          criterionL1, scheduler=scheduler, logger=logger)

    # Final test
    with open(config.model_path, 'rb') as f:
        model = torch.load(f)

    val_loss, val_corr, val_rse, val_rae = evaluate(model, data["valid_dataloader"], device, criterionMSE,
                                                    criterionL1, scales, rse_val_d, rae_val_d)
    test_loss, test_corr, test_rse, test_rae = evaluate(model, data["test_dataloader"], device, criterionMSE,
                                                        criterionL1, scales, rse_test_d, rae_test_d)

    if logger is not None:
        logger.info("final valid rse {:5.4f} | valid rae {:5.4f} | valid corr {:5.4f}".format(
            val_rse, val_rae, val_corr))
        logger.info("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(
            test_rse, test_rae, test_corr))
    return val_corr / val_rse


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    logger.info("Number of finished trials: ", len(study.trials))

    logger.info("Best trial:")
    trial = study.best_trial
    logger.info("  Value: ", trial.value)
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))
