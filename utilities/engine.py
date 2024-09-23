import math
import sys
import time
import torch

# downloaded from https://github.com/pytorch/vision/tree/main/references/detection
# codes in this file are referenced from this git repository
from utilities.helpers import convert_data_format
import utilities.utils as utils

# downloaded from https://github.com/pytorch/vision/tree/main/references/detection
from utilities.coco_eval import CocoEvaluator
from pycocotools.coco import COCO

from tqdm.notebook import tqdm


def train_one_epoch(
                model, 
                optimizer, 
                data_loader, 
                device, 
                epoch, 
                print_freq, 
                scaler=None, 
                tb_writer=None                
):
    """ train model in one epoch

    :param model: model to be trained
    :param optimizer: optimizer used for training, defined in torch.optim
    :param data_loader: DataLoader object to load data
    :param device: which gpu device is used for training
    :param epoch: current epoch index
    :param print_freq: frequency of info-printing
    :param scaler: scale gradient, defined in torch.cuda.amp.GradScaler
    :param tb_writer: tensorboard_SummaryWriter to write training info
    :return metric_logger: message logged
    """

    # set model in train mode
    model.train()
    
    # TODO: MetricLogger
    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # # loss value when evaluating
    running_loss = 0.0

    for batch_id, batch_data in enumerate(
                tqdm(
                    metric_logger.log_every(data_loader, print_freq, header), 
                    total=len(data_loader)
                )
        ):
        # adjust data in correct layout
        if len(batch_data) == 0:
            print('No Data in this Batch!')
            continue
        images, targets = batch_data

        # AssertionError: All bounding boxes should have positive height and width.
        heights = (targets[0]['boxes'][:, :, 3] - targets[0]['boxes'][:, :, 1]).flatten()
        widths = (targets[0]['boxes'][:, :, 2] - targets[0]['boxes'][:, :, 0]).flatten()
        if any(heights <= 0) or any(widths <= 0):
            continue
            
        images, targets = convert_data_format(images, targets, device)
        
        # eanble auto mixed precision (amp) to save memory and speed up training
        # only available when gpu is available
        # within autocast, only forward-propagation (back-prop cannot be inside!!)
        with torch.amp.autocast(str(device), enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        running_loss += loss_value

        # if Nan loss value, print info and exit program
        if not math.isfinite(loss_value):
            print(f'Loss is {loss_value}, Stopping Training!!')
            print(loss_dict_reduced)
            print(targets)
            sys.exit(1)

        optimizer.zero_grad()

        # scale loss values in case gradient vanish
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if tb_writer is not None:
            tb_x = epoch * len(data_loader) + batch_id + 1
            tb_writer.add_scalar('Loss/Train', loss_value, tb_x)

        metric_logger.update(loss_value=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # average loss per batch
    avg_loss = running_loss / len(data_loader)

    return avg_loss, metric_logger
    

@torch.inference_mode()
def evaluate(model, data_loader, device, print_freq):
    """ evaluate model

    :param model: model to be evaluated
    :param dataloader: DataLoader object to load data
    :param device: which device
    :param print_freq: freq of info-printing
    :return metric_logger: message logged
    """

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # loss value when evaluating
    running_loss = 0.0

    for batch_id, batch_data in enumerate(
                tqdm(
                    metric_logger.log_every(data_loader, print_freq, header), 
                    total=len(data_loader)
                )
        ):
        # adjust data in correct layout
        images, targets = batch_data

        # AssertionError: All bounding boxes should have positive height and width.
        heights = (targets[0]['boxes'][:, :, 3] - targets[0]['boxes'][:, :, 1]).flatten()
        widths = (targets[0]['boxes'][:, :, 2] - targets[0]['boxes'][:, :, 0]).flatten()
        if any(heights <= 0) or any(widths <= 0):
            continue
            
        images, targets = convert_data_format(images, targets, device)

        # start time point
        model_time = time.time()

        # compute loss
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        # update running_loss
        running_loss += loss_value

        # end time point
        model_time = time.time() - model_time

        # update losses
        metric_logger.update(loss_value=loss_value)
        metric_logger.update(model_time=model_time)

    # average loss per batch
    avg_loss = running_loss / len(data_loader)

    return avg_loss, metric_logger


@torch.inference_mode()
def evaluate_accuracy(
                model, 
                data_loader, 
                annotation_filepath, 
                device,
                print_freq,
):

    # set model in evaluation mode
    model.eval()
    # set metrid_logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test[Accuracy]:"

    # set coco evaluator
    coco = COCO(annotation_filepath)
    coco_evaluator = CocoEvaluator(coco, ['bbox', 'keypoints'])

    for batch_id, batch_data in enumerate(
                tqdm(
                    metric_logger.log_every(data_loader, print_freq, header), 
                    total=len(data_loader)
                )
        ):
        # adjust data in correct layout
        images, targets = batch_data
        images = images.to(device)
        for val in targets[0].values():
            val = val.to(device)
        # images, targets = convert_data_format(images, targets, device)


        # compute outputs
        # start time point
        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
        res = {target['image_id'].item(): output for target, output in zip(targets, outputs)}
        # end time point
        model_time = time.time() - model_time

        # update results
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator, metric_logger
    














        
    