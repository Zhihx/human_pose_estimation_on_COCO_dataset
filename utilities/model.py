from math import sqrt
import torch
import torch.nn as nn
import torchvision


class FSRCNN(nn.Module):
    """ Fast super resolution convolutional neural network
       :param: upscale_factor (int): Image magnification factor.
    """

    def __init__(self, upscale_factor: int) -> None:
        super(FSRCNN, self).__init__()
        # Feature extraction layer.
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, 56, (5, 5), (1, 1), (2, 2)),
            nn.PReLU(56)
        )

        # Shrinking layer.
        self.shrink = nn.Sequential(
            nn.Conv2d(56, 12, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(12)
        )

        # Mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12)
        )

        # Expanding layer.
        self.expand = nn.Sequential(
            nn.Conv2d(12, 56, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(56)
        )

        # Deconvolution layer.
        self.deconv = nn.ConvTranspose2d(56, 1, (9, 9), (upscale_factor, upscale_factor), (4, 4),
                                         (upscale_factor - 1, upscale_factor - 1))

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature_extraction(x)
        out = self.shrink(out)
        out = self.map(out)
        out = self.expand(out)
        out = self.deconv(out)

        return out

    # The filter weight of each layer is
    # a Gaussian distribution with zero mean and standard deviation
    # initialized by random extraction 0.001 (deviation is 0).
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

        nn.init.normal_(self.deconv.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.deconv.bias.data)


class Model(nn.Module):
    def __init__(
            self,
            num_input_channels: int,
            num_measurements: int,
            image_height: int,
            image_width: int,
            upscale_factor: int,
    ) -> None:
        super(Model, self).__init__()

        self.num_input_channels = num_input_channels
        self.num_measurements = num_measurements
        self.image_height = image_height
        self.image_width = image_width
        self.upscale_factor = upscale_factor

        self.modules_before_keypoint_module = build_modules_before_keypoint_module(
            self.num_input_channels,
            self.num_measurements,
            self.image_height,
            self.image_width,
            self.upscale_factor
        )

        self.keypoint_module = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            weights=None,
            num_keypoints=17
        )

    @staticmethod
    def convert_image_layout(tensor_nchw):
        """ convert a tensor with NCHW layout (batch, channel, height, width)
            to a list with multiple CHW (channel, height, width) tensors

        :param tensor_nchw: tensor with (batch, channel, height, width)
        :return list_tensor_chw: a list containing tensors with (channel, height, width) layout
        """
        # number of images in a batch
        batch_size = tensor_nchw.shape[0]
        # extract each tensor with chw layout and append it to a list
        list_tensor_chw = [tensor_nchw[i, :, :, :] for i in range(batch_size)]
        return list_tensor_chw

    @staticmethod
    def convert_target_layout(batch_targets):
        """ convert a list with only one dictionary as its element
            values inside the dictionary are with (batch, ...) layout

        :param batch_targets: a list with only one dictionary as its element
            values inside the dictionary are with (batch, ...) layout
        :return list_target: a list with dictionaries, each of dictionaries for an image
        """
        list_target = []
        for i in range(batch_targets[0]['keypoints'].shape[0]):
            list_target.append({
                'keypoints': batch_targets[0]['keypoints'][i, :, :, :].reshape(1, 17, 3),
                'boxes': batch_targets[0]['boxes'][i, :, :].reshape(1, 4),
                'labels': batch_targets[0]['labels'][i, :].reshape(1)
            })
        return list_target

    def forward_eval(self, x):
        """ forward function when model in eval mode

        :param x: image data
        :return output_eval: output annotations
        """

        feature_map = self.modules_before_keypoint_module(x)
        feature_map_ = self.convert_image_layout(feature_map)
        output_eval = self.keypoint_module(feature_map_)
        return output_eval

    def forward_train(self, x, batch_targets):
        """ forward function when model in train mode

        :param x: image data
        :param batch_targets: ground truth annotations
        :return output_train: loss information
        """

        batch_targets_ = self.convert_target_layout(batch_targets)
        feature_map = self.modules_before_keypoint_module(x)
        feature_map_ = self.convert_image_layout(feature_map)
        output_train = self.keypoint_module(feature_map_, batch_targets_)
        return output_train

    def forward(self, x, *args):
        """ forward function

        :param x: image data
        :param args: ground truth annotations (optional dependent on mode of model)
        :return:
        """
        if self.training:
            # when model in train mode
            batch_targets = args[0]
            output_train = self.forward_train(x, batch_targets)
            return output_train
        else:
            # when model in eval mode
            output_eval = self.forward_eval(x)
            return output_eval


def build_modules_before_keypoint_module(
        num_input_channels: int,
        num_measurements: int,
        image_height: int,
        image_width: int,
        upscale_factor: int
):
    # encoder: define components of encoder
    encoder_illumination_module = nn.Conv2d(
        in_channels=num_input_channels,
        out_channels=num_measurements,
        kernel_size=(image_height, image_width)
    )
    encoder_flatten_module = nn.Flatten()
    encoder_linear_module = nn.Linear(
        num_measurements,
        int((image_height * image_width) / 4)
    )
    # encoder: combine all encoder modules together
    encoder_module = nn.Sequential(
        encoder_illumination_module,
        encoder_flatten_module,
        encoder_linear_module
    )

    # decoder: define components of decoder
    decoder_reshape_module = nn.Unflatten(
        dim=1,
        unflattened_size=(num_input_channels, int(image_height / 2), int(image_width / 2))
    )
    decoder_FSRCNN_module = FSRCNN(upscale_factor=upscale_factor)
    # decoder: combine all decoder modules together
    decoder_module = nn.Sequential(
        decoder_reshape_module,
        decoder_FSRCNN_module,
    )

    # combine all components together
    modules_before_keypoint_module = nn.Sequential(
        encoder_module,
        decoder_module,
    )

    return modules_before_keypoint_module

