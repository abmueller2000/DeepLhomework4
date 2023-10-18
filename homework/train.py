import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = Detector().to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_data = load_detection_data('dense_data/train')
    valid_data = load_detection_data('dense_data/valid')

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        for img, gt_det in train_data:
            img, gt_det = img.to(device), gt_det.to(device)

            pred_det = model(img)
            loss_val = criterion(pred_det, gt_det)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
                log(train_logger, img, gt_det, pred_det, global_step)  # logging images and heatmaps

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        # validation and logging for validation can go here

        save_model(model)


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('-n', '--num_epoch', type=int, default=75)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2)
    parser.add_argument('-c', '--continue_training', action='store_true')

    args = parser.parse_args()
    train(args)
