import os
import logging
import time
from datetime import timedelta
import pandas as pd
import shutil


from tensorboardX import SummaryWriter
import os
import torch

class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ""


def create_logger(filepath, rank):
    """
    Create a logger.
    Use a different log file for each process.
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        if rank > '0':
            filepath = "%s-%i" % (filepath, rank)
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    return logger


class Recoder:
    def __init__(self):
        self.metrics = {}

    def record(self, name, value):
        if name in self.metrics.keys():
            self.metrics[name].append(value)
        else:
            self.metrics[name] = [value]

    def summary(self):
        kvs = {}
        for key in self.metrics.keys():
            kvs[key] = sum(self.metrics[key]) / len(self.metrics[key])
            del self.metrics[key][:]
            self.metrics[key] = []
        return kvs


class Logger:
    def __init__(self, args):
        self.writer = SummaryWriter(args.tensorboard_path)
        self.recoder = Recoder()
        self.model_dir = args.model_dir
        self.logger = create_logger(
            os.path.join(args.model_dir, "train.log"), rank='0'
        )

    def tensor2img(self, tensor):
        # implement according to your data, for example call viz.py
        return tensor.detach().cpu().numpy()

    def record_scalar(self, name, value):
        self.recoder.record(name, value)

    def save_curves(self, epoch):
        kvs = self.recoder.summary()
        for key in kvs.keys():
            self.writer.add_scalar(key, kvs[key], epoch)

    def save_imgs(self, names2imgs, epoch):
        for name in names2imgs.keys():
            self.writer.add_image(name, self.tensor2img(names2imgs[name]), epoch)

    def update_checkpoint(self, model, optimizer, epoch):
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(
            save_dict,
            os.path.join(self.model_dir, "checkpoint.pth.tar"),
        )
        # don't save model, which depends on python path
        # save model state dict
    def save_checkpoint(self, model, optimizer, epoch):
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(
            save_dict,
            os.path.join(self.model_dir, "checkpoint.pth.tar"),
        )
        # don't save model, which depends on python path
        # save model state dict
        shutil.copyfile(
            os.path.join(self.model_dir, "checkpoint.pth.tar"),
            os.path.join(self.model_dir, "ckp-" + str(epoch) + ".pth"),
        )
        


class PD_Stats(object):
    """
    Log stuff with pandas library
    """

    def __init__(self, path, columns):
        self.path = path

        # reload path stats
        if os.path.isfile(self.path):
            self.stats = pd.read_pickle(self.path)

            # check that columns are the same
            assert list(self.stats.columns) == list(columns)

        else:
            self.stats = pd.DataFrame(columns=columns)

    def update(self, row, save=True):
        self.stats.loc[len(self.stats.index)] = row

        # save the statistics
        if save:
            self.stats.to_pickle(self.path)