import cv2
import matplotlib
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from typing import List, Dict, Union

edges = [
    (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
    (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
    (12, 14), (14, 16), (5, 6)
]


def draw_keypoints(
        annotations: List[Dict],
        image: Union[np.ndarray, torch.Tensor]
):
    """ draw keypoints on an image

    :param annotations: annotations from COCO dataset or from a Keypoint_RCNN Model
    :param image: image
    :return: image_copy with keypoints drawn
    """
    image_copy = None
    if isinstance(image, torch.Tensor):
        if len(image.shape) == 4:
            image_copy = image.detach().squeeze(0).permute(1, 2, 0).numpy().copy()
        elif len(image.shape) == 3:
            image_copy = image.detach().permute(1, 2, 0).numpy().copy()
            if image_copy.shape[2] == 1:
                image_copy = image_copy[:,:,0]
    elif isinstance(image, np.ndarray):
        image_copy = image.copy()
    else:
        raise TypeError("WRONG TYPE OF INPUT IMAGE!")
                
    if 'scores' in annotations[0].keys():
        keypoints = annotations[0]['keypoints'].cpu().detach().numpy()
        scores = annotations[0]['scores'].cpu().detach().numpy()
        for i in range(keypoints.shape[0]):
            # proceed to draw the keypoints if the confidence score is above 0.9
            if scores[i] > 0.9:
                keypoints_one_instance = keypoints[i, :, :].reshape(-1, 3)
                for p in range(keypoints_one_instance.shape[0]):
                    # draw the keypoints
                    if len(image_copy.shape) == 3:
                        cv2.circle(image_copy, (int(keypoints_one_instance[p, 0]), int(keypoints_one_instance[p, 1])),
                                   3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                    else:
                        cv2.circle(image_copy, (int(keypoints_one_instance[p, 0]), int(keypoints_one_instance[p, 1])),
                                   3, (1., 1., 1.), thickness=-1, lineType=cv2.FILLED)
                for ie, e in enumerate(edges):
                    # get different colors for the edges
                    if len(image_copy.shape) == 3:
                        rgb = matplotlib.colors.hsv_to_rgb([
                            ie / float(len(edges)), 1.0, 1.0
                        ])
                        rgb = rgb * 255
                    else:
                        # if grayscale image, draw edges using white color
                        rgb = [1., 1., 1.]
                    # join the keypoint pairs to draw the skeletal structure
                    cv2.line(image_copy, (int(keypoints_one_instance[e, 0][0]), int(keypoints_one_instance[e, 1][0])),
                             (int(keypoints_one_instance[e, 0][1]), int(keypoints_one_instance[e, 1][1])),
                             tuple(rgb), 2, lineType=cv2.LINE_AA)
            else:
                continue
    else:
        keypoints = annotations[0]['keypoints'].cpu().detach().numpy()
        for i in range(keypoints.shape[0]):
            invalid_keypoints = []
            keypoints_one_instance = keypoints[i, :, :].reshape(-1, 3)
            # proceed to draw keypoints if the visibility is not 0 (visible or labelled)              
            for j in range(17):
                keypoint = keypoints_one_instance[j, :]
                if not keypoint[-1] == 0:
                    # draw the keypoints
                    if len(image_copy.shape) == 3:
                        cv2.circle(image_copy, (int(keypoint[0]), int(keypoint[1])),
                                   1, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                    else:
                        cv2.circle(image_copy, (int(keypoint[0]), int(keypoint[1])),
                                   1, (0, 0, 1.), thickness=-1, lineType=cv2.FILLED)
                else:
                    # add keypoints that are not visible and labelled in invalid_keypoints
                    invalid_keypoints.append(j)
                    continue
            for ie, e in enumerate(edges):
                # get different colors for the edges
                if not len(list(set(e) & set(invalid_keypoints))):
                    if len(image_copy.shape) == 3:
                        rgb = matplotlib.colors.hsv_to_rgb([
                            ie / float(len(edges)), 1.0, 1.0
                        ])
                        rgb = rgb * 255
                    else:
                        # for grayscale image, use white color to draw edges
                        rgb = (1., 1., 1.)
                    # join the keypoint pairs to draw the skeletal structure
                    cv2.line(image_copy,
                             (
                                 int(keypoints_one_instance[e, 0][0]),
                                 int(keypoints_one_instance[e, 1][0])),
                             (
                                 int(keypoints_one_instance[e, 0][1]),
                                 int(keypoints_one_instance[e, 1][1])),
                             tuple(rgb), 1, lineType=cv2.LINE_AA
                             )
                else:
                    continue
    return image_copy

def convert_data_format(batch_images, batch_targets, device):
    """ convert data format

    :param batch_images: [N, C, H, W] tensor, N means batch_size (number of images)
    :param batch_targets: a list with a dictionary where values are with layouts of [N, ...]
    :param device: move data to which device 
    :return images_: a list with N elements, each element is a [C, H, W] tensor
    :return targets_: a list with N dictionaries, each element contains anntations in an image
    """
    images_ = []
    targets_ = []

    for i in range(batch_images.shape[0]):
        images_.append(batch_images[i, :, :, :].to(device))
        targets_.append({
            'keypoints': batch_targets[0]['keypoints'][i, :, :, :].to(device),
            'boxes': batch_targets[0]['boxes'][i, :].to(device),
            'labels': batch_targets[0]['labels'][i].to(device),
            'image_id': batch_targets[0]['image_id'][i]
        })
        
    return images_, targets_

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0: return torch.Tensor()
    return default_collate(batch)