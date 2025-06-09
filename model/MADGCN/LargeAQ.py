import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from easydict import EasyDict
from basicts.data import TimeSeriesForecastingDataset
from basicts.losses import masked_mae

from model.MADGCN.arch.arch import MADGCN
from model.MADGCN.arch.runner import MADGCNRunner

CFG = EasyDict()

CFG.DESCRIPTION = "MADGCN model configuration for LargeAQ dataset"
CFG.RUNNER = MADGCNRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "LargeAQ"
CFG.DATASET_TYPE = "AirQuality"
CFG.DATASET_INPUT_LEN = 96
CFG.DATASET_OUTPUT_LEN = 96
CFG.GPU = torch.cuda.is_available()
CFG.NULL_VAL = 0.0

CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "MADGCN"
CFG.MODEL.ARCH = MADGCN

adj_mx_file = "datasets/LargeAQ/adj_mx.pkl"

CFG.MODEL.PARAM = {
    "num_nodes": 1341,
    "in_dim": 1,
    "out_dim": 1,
    "embed_dim": 64,
    "gcn_depth": 2,
    "seq_length": 96,
    "horizon": 96,
    "layers": 2,
    "patch_len": 16,
    "stride": 8,
    "d_model": 128,
    "mixer_kernel_size": 8,
    "predefined_A": adj_mx_file,
    "predefined_CAG": adj_mx_file,
    "fusion_alpha": 0.7,
    "cycle_len": 24,
    "gamma": 0.1,
    "dropout": 0.3,
}

CFG.MODEL.FORWARD_FEATURES = [0]
CFG.MODEL.TARGET_FEATURES = [0]

CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.02,
    "weight_decay": 0.004,
}

CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [50, 150, 250],
    "gamma": 0.5
}

CFG.TRAIN.CLIPGRAD = EasyDict()
CFG.TRAIN.CLIPGRAD.ENABLED = True
CFG.TRAIN.CLIPGRAD.CLIPNORM = 5.0

CFG.TRAIN.NUM_EPOCHS = 300
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)

CFG.TRAIN.EARLY_STOPPING = EasyDict()
CFG.TRAIN.EARLY_STOPPING.ENABLED = True
CFG.TRAIN.EARLY_STOPPING.PATIENCE = 20
CFG.TRAIN.EARLY_STOPPING.MONITOR = "val_loss"
CFG.TRAIN.EARLY_STOPPING.MODE = "min"

CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.TRAIN.DATA.BATCH_SIZE = 16
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 4
CFG.TRAIN.DATA.PIN_MEMORY = False

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.VAL.DATA.BATCH_SIZE = 16
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 4
CFG.VAL.DATA.PIN_MEMORY = False

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.TEST.DATA.BATCH_SIZE = 16
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 4
CFG.TEST.DATA.PIN_MEMORY = False

CFG.DATA_SPLIT = EasyDict()
CFG.DATA_SPLIT.TRAIN_RATIO = 0.5
CFG.DATA_SPLIT.VAL_RATIO = 0.25
CFG.DATA_SPLIT.TEST_RATIO = 0.25
CFG.DATA_SPLIT.TEMPORAL_ORDER = True

CFG.DATA_NORM = EasyDict()
CFG.DATA_NORM.TYPE = "zscore"
CFG.DATA_NORM.USE_TRAIN_STATS = True 