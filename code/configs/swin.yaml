from yacs.config import CfgNode as CN

config = CN()

config.DATA = CN()
config.DATA.IMG_SIZE = 224
# config.DATA_PATH = _KVASIR_CAPSULE
config.DATA.BATCH_SIZE = 8
config.DATA.NUM_WORKERS = 2

config.MODEL = CN()
config.MODEL.NUM_CLASSES = 14
# config.MODEL.TYPE = 'coat'

config.TRAIN = CN()
# config.TRAIN.DEVICE = str(device)
# config.TRAIN.CLASSES_WEIGHT = True
config.TRAIN.MARGIN = 'arcface'
config.TRAIN.START_EPOCH = 0
config.TRAIN.EPOCHS = 300
config.TRAIN.BASE_LR = 1e-3
config.TRAIN.WARMUP_LR = 5e-7
config.TRAIN.WEIGHT_DECAY = 1e-5
config.TRAIN.WARMUP_EPOCHS = 20

config.TRAIN.OPTIMIZER = CN()
config.TRAIN.OPTIMIZER.LOSS_TYPE = 'ce'
config.TRAIN.OPTIMIZER.NAME = 'sgd'
config.TRAIN.OPTIMIZER.EPS = 1e-8
config.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
config.TRAIN.OPTIMIZER.MOMENTUM = 0.9
config.TRAIN.OPTIMIZER.FOCAL_LOSS = False

config.TRAIN.LR_SCHEDULER = CN()
# config.TRAIN.LR_SCHEDULER.NAME = 'cosine'
config.TRAIN.LR_SCHEDULER.STEP_SIZE = 10
config.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.85