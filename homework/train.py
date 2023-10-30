import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from .dense_transforms import Compose, ToHeatmap, ToTensor

import torch.utils.tensorboard as tb


def train(args):
    from os import path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Detector(num_classes=3).to(device)
    transform = Compose([ToTensor(), ToHeatmap()])

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_data = load_detection_data('dense_data/train', transform=transform)
    valid_data = load_detection_data('dense_data/valid')

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        for batch_idx, (img, gt_det, _) in enumerate(train_data):
            img, gt_det = img.to(device), gt_det.to(device)
            
            optimizer.zero_grad()
            pred_det = model(img)
            loss_val = criterion(pred_det, gt_det)

            # if train_logger is not None:
            #     train_logger.add_scalar('loss', loss_val, global_step)
            #     log(train_logger, img, gt_det, pred_det, global_step)  # logging images and heatmaps

            loss_val.backward()
            optimizer.step()
            
            global_step+=1
            if batch_idx % args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(img)}/{len(train_data.dataset)} '
                    f'({100. * batch_idx / len(valid_data):.0f}%)]\tLoss: {loss_val.item():.6f}')

        # validation and logging for validation can go here

    torch.save(model.state_dict(), 'det.th')


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

    parser.add_argument('-n', '--num_epochs', type=int, default=35, help='Number of epochs')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--log-interval', type=int, default=10, help='Num of batches to wait before logging training status')
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor()])')
    parser.add_argument('--log_dir')

    args = parser.parse_args()
    train(args)
