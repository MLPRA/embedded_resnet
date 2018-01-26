import chainer
import numpy as np
import matplotlib.pyplot as plt

from src.dataset import ImageSegment, LabelHandler
from src.resnet import ResNet49Layers

import chainer


def predict(model_path: "path to resnet", image:(str, "path of image"), label_names: "path to label names file",
            xmin: "Minimum x value of slice", xmax: "Maximum x value of slice",
            ymin: "Minimum y value of slice", ymax: "Maximum y value of slice",
            gpu: "gpu id, negative for cpu" = 0):

    resnet = ResNet49Layers(pretrained_model=model_path)

    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        resnet.to_gpu()

    label_handler = LabelHandler(label_names)

    image_segment = ImageSegment(image, xmin, ymin, xmax, ymax)
    image = image_segment()

    prediction = resnet.predict([image], oversample=True)[0].array

    for i in range(len(prediction)):
        if prediction[i] > 0:
            output = '{}: {}%'.format(label_handler.get_label_str(i), prediction[i] * 100)
            print(output)

    return 0