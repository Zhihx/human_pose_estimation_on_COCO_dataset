import os
from pathlib import Path
import numpy as np
import skimage.io as io
from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import v2

from typing import Any, Callable, Optional, Tuple, List, Dict, Union

class COCOKeypoint(Dataset):
    """ A torch-based Dataset to access data from COCO Keypoint Dataset

    :param root: string (or PosixPath object in pathlib package) 
                    where COCO dataset is stored
    :param annotation_file: string (or PosixPath object in pathlib package) 
                    where annotation file (.json file) is stored
    :param transform: a function to transform images
    :param target_transform: a function to transform annotations (target)
    :param is_cropped: whether images are cropped to a specific size
    :param crop_size: effective if is_cropped is True
    :param is_grayscale: whether images are converted into grayscale images
    :returns: image, target (image and its annotations)
    """
    
    def __init__(
            self,
            root: Union[str, Path],
            annotation_filepath: Union[str, Path],
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            is_cropped = False,
            crop_size = None,
            is_grayscale = False
    ) -> None:
        
        super(COCOKeypoint, self).__init__()

        # root directory of COCO dataset
        self.root = root 
        # path of annotation file
        self.annotation_filepath = annotation_filepath 
        # whether the images are cropped 
        self.is_cropped = is_cropped
        # crop_size
        self.crop_size = crop_size
        # whether the images are converted into grayscale images
        self.is_grayscale = is_grayscale

        if transform is None:
            self.transform = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(), 
                                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]) # not additional transform on images
        else:
            self.transform = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(), 
                                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                transform
            ]) # additional transform on images           

        if target_transform is not None:
            # transform on targets (annotations)
            self.target_transform = target_transform 

        self.coco = COCO(self.annotation_filepath) # define a coco object to fetch data from .json file

        # fetch image IDs corresponding to category 'person'
        category_ids = self.coco.getCatIds(catNms=['person'])
        self.image_ids = list(sorted(self.coco.getImgIds(catIds=category_ids)))

    
    def _load_image(self, id: int) -> torch.Tensor:
        """ load and transform image

        :param id: image_id
        :return: image data (torch.FloatTensor)
        """
        try:
            # load local file 
            filepath = self.coco.loadImgs(id)[0]["file_name"]
            img_temp = io.imread(os.path.join(self.root, filepath))
        except FileNotFoundError:
            # if local file not found, load using url
            print(f'FileNotFound: {filepath}')
            url = self.coco.loadImgs(id)[0]["coco_url"]
            img_temp = io.imread(url)

        # if img_temp is HWC layout, transform to CHW layout
        # if img_temp is uint8 ([0-255]), transform to [0.0-1.0]
        # default data type is torch.float32, 
        # .type(torch.float32) is unnecessary
        img = self.transform(img_temp)
        
        return img

    
    def _load_target(self, id: int) -> List[Any]:
        """ load target (annotations)

        :param id: image_id
        :return: target (annotations), 
                    a list with only one dictionary containing all annotations on an image
        """

        annotation_temp = self.coco.loadAnns(self.coco.getAnnIds(id))
        keypoints_temp = np.zeros((len(annotation_temp), 17, 3))
        boxes_temp = np.zeros((len(annotation_temp), 4))
        labels_temp = np.zeros((len(annotation_temp)))

        for i in range(len(annotation_temp)):
            keypoints_temp[i, :, :] = np.reshape(annotation_temp[i]['keypoints'], (17, 3))
            boxes_temp[i, :] = self._convert_boxes_from_bbox(annotation_temp[i]['bbox'])
            labels_temp[i] = annotation_temp[i]['category_id']

        # a list with only one dictionary containing all annotations on an image
        target = [
            {
                'keypoints': torch.FloatTensor(keypoints_temp),
                'boxes': torch.FloatTensor(boxes_temp),
                'labels': torch.tensor(labels_temp, dtype=torch.int64),
                'image_id': id
            }
        ]
        
        return target

        
    @staticmethod
    def _convert_boxes_from_bbox(bbox: List):
        """ convert roi box style in COCO dataset to roi box style in Keypoint R-CNN

        :param bbox: a 4-elements list to represent box roi style in COCO dataset
        :return: a (1, 4) ndarray boxes to represent box roi in Keypoint R-CNN
        """

        # bbox in coco dataset: [x1, y1, width, height]
        # boxes in output of NN: [x1, y1, x2, y2]
        bbox = np.reshape(bbox, (1, 4))
        boxes = bbox.copy()
        boxes[0, 2] += boxes[0, 0]
        boxes[0, 3] += boxes[0, 1]
        
        return boxes

        
    @staticmethod
    def _transform_image_ane_annotations(
            image: torch.FloatTensor,
            target: List[Dict],
            size: int,
    ):
        # Obtain boxes for all instances 
        # (Size: N x 4, N is the number of instances)
        boxes = target[0]['boxes']
        # Obtain keypoints for all instances 
        # (Size: N x 17 x 3, N is the number of instances)
        keypoints = target[0]['keypoints']
        # Obtain labels for all instances
        labels = target[0]['labels']
        # Obtain boxes of instances with non-all-zero keypoints  
        boxes_effective = boxes[torch.where(keypoints[:,:,2].any(axis=1))[0]]
        # Obtain keypoints of instances with non-all-zero keypoints
        keypoints_effective = keypoints[torch.where(keypoints[:,:,2].any(axis=1))[0]]
        # Obtain labels of instances with non-all-zero keypoints
        labels_effective = labels[torch.where(keypoints[:,:,2].any(axis=1))[0]]
        
        # Find the largest box with keypoints
        if keypoints_effective.shape[0] == 0:
            # if non-effective keypoints, return (None, None)
            return (None, None)
        else:
            # else return effective image data and annotations
            id = torch.argmax(
                        (boxes_effective[:, 2] - boxes_effective[:, 0]) * (boxes_effective[:, 3] - boxes_effective[:, 1])
            )
        
        # Get the box with the largest area
        roi_box = boxes_effective[id, :]
        x1 = int(torch.floor(roi_box[0]))
        y1 = int(torch.floor(roi_box[1]))
        x2 = int(torch.ceil(roi_box[2]))
        y2 = int(torch.ceil(roi_box[3]))
        
        # Crop the image
        if len(image.shape) == 3:
            image_ = image[:, y1:y2, x1:x2]
            _, h, w = image_.shape
        elif len(image.shape) == 2:
            image_ = image[y1:y2, x1:x2]
            h, w = image_.shape
            
        # Pad the image with same width and height
        if h > w:
            image_ = v2.Pad([0, 0, h - w, 0], 1.0)(image_)
        elif w > h:
            image_ = v2.Pad([0, 0, 0, w - h], 1.0)(image_)
        # Resize to specified size
        image_ = v2.Resize(size)(image_)

        
        # Adjust the coordinates of keypoints
        keypoints_temp = keypoints_effective[id, :, :]
        if len(torch.where(keypoints_temp.any(axis=1))[0]) < 4:
            # if number of keypoints less than 4
            # return (None, None)
            return (None, None)
        for i in range(17):
            if keypoints_temp[i, 2] != 0:
                keypoints_temp[i, 0] = (keypoints_temp[i, 0] - x1) * size / np.maximum(h, w)
                keypoints_temp[i, 1] = (keypoints_temp[i, 1] - y1) * size / np.maximum(h, w)
        keypoints_temp[:, 2] = keypoints_temp[:, 2]
        target_ = [{
            'keypoints': keypoints_temp.unsqueeze_(0),
            'boxes': torch.FloatTensor([0, 0, size - 1, size - 1]).reshape(1, 4),
            'labels': labels_effective[id].reshape(1),
            'image_id': target[0]['image_id']
        }]
        
        return image_, target_

    
    @staticmethod
    def _convert_image_to_grayscale(image):
    
        # Convert to grayscale
        image_grayscale = v2.Grayscale(num_output_channels=1)(image) 
        
        return image_grayscale
        
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """ obtain one item in the dataset

        :param index: index
        :return: image, target (image and its annotations with index)
        """
        id = self.image_ids[index]

        image = self._load_image(id)
        target = self._load_target(id)

        if self.is_grayscale:
            image = self._convert_image_to_grayscale(image)
        
        if self.is_cropped:
            image, target = self._transform_image_ane_annotations(image, target, self.crop_size)

        return image, target
        

    def __len__(self):
        """ return length of the dataset

        :return: length of the dataset
        """
        return len(self.image_ids)