"""Example OctoAI service scaffold: YOLOv8."""
import json
from typing import List

import numpy as np
from PIL import Image as PImage
from pydantic import BaseModel, Field
from ultralytics import YOLO

from octoai.service import Service
from octoai.types import Image

"""
YOLO (You Only Look Once) is a real-time object detection algorithm.
YOLO works by dividing an image into a grid of cells. Each cell predicts
the probability of each object class being present in the cell, as well
as the bounding box coordinates of the object.

The `YOLOv8Service.infer()` method returns a `YOLOResponse`, which represents
a list of `Detections`. Each `Detection` represents a bounding box with a
class label and confidence value. A `YOLOResponse` also contains an `Image`
object, which represets a serialized binary image file encoded as base64
that has the detection boxes drawn in it. See also the comments inside
`YOLOv8Service.infer()` below.

The YOLOv8 model returns a list of detections that look like this:

    [{
        'name': 'bus',
        'class': 5,
        'confidence': 0.95,
        'box': {
            'x1': 2.91,
            'x2': 809.51,
            'y1': 230.68,
            'y2': 881.00
        }
    }, ...]

For more information about defining your own entities, such as `YOLOResponse`,
`Detection` and `Box` as in this example, see:
https://docs.octoai.cloud/docs/containerize-your-code-with-our-cli#appendix-openapi-specification-and-pydantic-types

"""


class Box(BaseModel):
    """Represents corners of a detection box."""

    x1: float
    x2: float
    y1: float
    y2: float


class Detection(BaseModel):
    """Represents a detection."""

    name: str
    class_: int = Field(..., alias="class")
    confidence: float
    box: Box


class YOLOResponse(BaseModel):
    """Response includes list of detections and rendered image."""

    detections: List[Detection]
    image_with_boxes: Image


class YOLOv8Service(Service):
    """An OctoAI service extends octoai.service.Service."""

    # Uncomment this function if you want to place large assets
    # (checkpoints, models, etc) on a docker volume instead of building
    # them into the image. This can provide faster
    # cold-start time. You may need to modify your inference pipeline
    # so that it leverages the locally-stored assets instead of
    # downloading them at container boot time.
    # def store_assets(self) -> None:
    #     """Download model assets."""
    #     # By default, instantiate the model to fill local disk
    #     # caches on the OctoAI volume. You can customize this if
    #     # your model can be fetched without instantiating it.
    #     self.setup()

    def setup(self):
        """Download model weights to disk."""
        self.model = YOLO("yolov8l.pt")

    def infer(self, image: Image) -> YOLOResponse:
        """Perform inference with the model.

        The `Image` type is a wrapper for binary image files that provides a
        consistent API specification for your endpoint. The image data is
        transferred over HTTP encoded as base64. This type provides support for
        encoding and decoding, reading image data as Pillow, creating image files
        from Pillow, and reading image files from disk or remote URLs.

        See also the API reference at
        https://octoml.github.io/octoai-python-sdk/octoai.html#module-octoai.types

        The `YOLOResponse` class is transferred over HTTP as JSON as described
        above.
        """
        image_pil = image.to_pil()
        output = self.model(image_pil)

        # no Python dict available directly from the model
        detections = json.loads(output[0].tojson())

        # get the rendered image with bounding boxes and labels
        # - Result.plot() generates BGR; flip to RGB
        img_out_numpy = np.flip(output[0].plot(), axis=2)
        img_out_pil = PImage.fromarray(img_out_numpy)

        # Return detection data and a rendered image with boxes
        return YOLOResponse(
            detections=[Detection(**d) for d in detections],
            image_with_boxes=Image.from_pil(img_out_pil),
        )
