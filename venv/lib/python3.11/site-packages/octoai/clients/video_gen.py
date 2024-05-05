"""OctoAI Video Generation."""
from __future__ import (
    annotations,  # required to allow 3.7+ python use type | syntax introduced in 3.10
)

from enum import Enum
from typing import Dict, List

from clients.image_gen.models import ImageEncoding, Scheduler, SDXLStyles
from octoai.client import Client
from octoai.clients.asset_orch import Asset
from octoai.errors import OctoAIValidationError
from octoai.types import Image, Video

SVD_ALLOWABLE_WIDTH_TO_HEIGHT = {
    1024: {576},
    576: {1024},
    768: {768},
}


MAX_CFG_SCALE = 10
MAX_STEPS = 50
MAX_MOTION_SCALE = 5
MAX_NOISE_AUG_STRENGTH = 1
MAX_NUM_VIDEOS = 16
MAX_FPS = 25


class Engine(str, Enum):
    """Engines."""

    SVD = "svd"

    def to_name(self):
        if self == self.SVD:
            return "Stable Video Diffusion"


class VideoGenerateResponse:
    """
    Video generation response.

    Contains a list of videos as well as a counter of those filtered for safety.
    """

    def __init__(self, videos: List[Video], removed_for_safety: int):
        self._videos = videos
        self._removed_for_safety = removed_for_safety

    @property
    def videos(self) -> List[Video]:
        """Return list of :class:`Video` generated from request."""
        return self._videos

    @property
    def removed_for_safety(self) -> int:
        """Return int representing number of videos removed for safety."""
        return self._removed_for_safety


class VideoGenerator(Client):
    """Client for video generation."""

    def __init__(
        self,
        api_endpoint: str | None = None,
        *args,
        **kwargs,
    ):
        if not api_endpoint:
            api_endpoint = "https://image.octoai.run/"

        if not api_endpoint.endswith("/"):
            api_endpoint += "/"
        self.api_endpoint = api_endpoint
        super(VideoGenerator, self).__init__(*args, **kwargs)

        if self._httpx_client.headers.get("Authorization") is None:
            msg = (
                "Authorization is required.  Please set an `OCTOAI_TOKEN` "
                "environment variable, or pass your token to the client using "
                "`client = ImageGenerator(token='your-octoai-api-token')`"
            )
            raise OctoAIValidationError(msg)

    # Raises OctoAIValidationError on failure to validate variable.
    # Does not validate strings currently, though should once API is stable.
    # TODO: standardize error strings (once more input types are known).
    def _validate_inputs(
        self,
        image: str | Image,
        engine: Engine | str,
        height: int | None,
        width: int | None,
        cfg_scale: float | None,
        steps: int | None,
        motion_scale: float | None,
        noise_aug_strength: float | None,
        num_videos: int | None,
        fps: int | None,
        seed: int | List[int] | None,
    ):
        """Validate inputs."""
        engines = [e.value for e in Engine]
        if engine not in engines:
            raise OctoAIValidationError(
                f"engine set to {engine}.  Must be one of: {', '.join(engines)}."
            )

        # Check number values in range
        if cfg_scale is not None and ((0 > cfg_scale) or (MAX_CFG_SCALE < cfg_scale)):
            raise OctoAIValidationError(
                f"cfg_scale set to: {cfg_scale}. Allowable range is > 0 and <= {MAX_CFG_SCALE}."
            )

        if noise_aug_strength is not None and (
            0 > noise_aug_strength or noise_aug_strength > MAX_NOISE_AUG_STRENGTH
        ):
            raise OctoAIValidationError(
                f"noise_aug_strength set to: {noise_aug_strength}. Allowable range is "
                f">= 0 and <= {MAX_NOISE_AUG_STRENGTH}."
            )
        if num_videos is not None and not (0 <= num_videos <= MAX_NUM_VIDEOS):
            raise OctoAIValidationError(
                f"num_images set to: {num_videos}.  Allowable range is > 0 and <= {MAX_NUM_VIDEOS}."
            )
        if isinstance(seed, list):
            for each in seed:
                if each is not None and (0 > each or each >= 2**32):
                    raise OctoAIValidationError(
                        f"seed({seed}) contains {each}. Allowable range is >= 0 and "
                        f"< 2**32."
                    )
        if type(seed) == int and not (0 > seed or seed >= 2**32):
            raise OctoAIValidationError(
                f"seed set to: {seed}.  Allowable range is >= 0 and < 2**32."
            )
        if steps is not None and not (0 <= steps <= MAX_STEPS):
            raise OctoAIValidationError(
                f"steps set to: {steps}.  Allowable range is > 0 and <= {MAX_STEPS}."
            )
        if fps is not None and not (1 <= fps <= MAX_FPS):
            raise OctoAIValidationError(
                f"fps set to: {fps}.  Allowable range is >= 1 and <= {MAX_FPS}."
            )
        if motion_scale is not None and not (0 <= motion_scale <= MAX_MOTION_SCALE):
            raise OctoAIValidationError(
                f"motion_scale set to: {motion_scale}.  Allowable range is >= 0 and <= {MAX_MOTION_SCALE}."
            )

        # Validate width and height to engine
        if (height is None) ^ (width is None):
            raise OctoAIValidationError(
                f"if height({height}) or width({width}) is set "
                f"to None, both must be None."
            )
        engine = Engine(engine)
        if height is not None and width is not None:
            self._validate_height_and_width_to_engine(engine, height, width)

        if not isinstance(image, Image):
            image = Image(image)  # type: ignore
        if not image.is_valid():
            msg = (
                "image is not a valid image.  May either use the "
                "octoai.types Image class or a base64 string."
            )
            raise OctoAIValidationError(msg)

    # Server will return a 500 error if incorrect height and width are entered
    @staticmethod
    def _validate_height_and_width_to_engine(engine: Engine, height: int, width: int):
        width_to_height_by_engine = {Engine.SVD.value: SVD_ALLOWABLE_WIDTH_TO_HEIGHT}

        # Set to correct const allowable values
        width_to_height = width_to_height_by_engine.get(engine)
        if width_to_height.get(width) is None:
            raise OctoAIValidationError(
                f"width ({width}): height ({height}) "
                f"values must match {engine.to_name()} allowable values or both be "
                f"None. Valid values for width are {list(width_to_height.keys())}."
            )
        if height not in width_to_height.get(width):
            raise OctoAIValidationError(
                f"width ({width}): height ({height}) "
                f"values must match {engine.to_name()} allowable values or both be "
                f"None. Valid width:height values are: {width_to_height}."
            )

    @staticmethod
    def _input_not_match_engine(name, value, engine, ok_engine):
        raise OctoAIValidationError(
            f"{name}({value}) is set but engine is set to "
            f"{engine}. {name} can only be used with {ok_engine}."
        )

    def generate(
        self,
        image: str | Image,
        engine: Engine | str,
        height: int | None = None,
        width: int | None = None,
        cfg_scale: float | None = 3.0,
        steps: int | None = 25,
        motion_scale: float | None = 0.5,
        noise_aug_strength: float | None = 0.02,
        num_videos: int | None = 1,
        fps: int | None = 7,
        seed: int | List[int] | None = None,
    ) -> VideoGenerateResponse:
        """
        Generate a list of videos based on request.

        :param engine: Required. "svd" for Stable Video Diffusion.
        :param image: Starting image. Requires a b64 string image or :class:`Image`.
        :param height: Height of video to generate, defaults to None.
        :param width: Width of video to generate, defaults to None.
        :param cfg_scale: How closely to adhere to image, defaults to 3.0.
            Must be >= 0 and <= 10.
        :param steps: How many steps of diffusion to run, defaults to 25.
            May be > 0 and <= 50.
        :param motion_scale: A floating point number between 0 and 5
            indicating how much motion should be in the generated video/animation.
        :param num_videos: How many videos to generate, defaults to 1.
            May be > 0 and <= 16.
        :param fps: How fast the generated frames should play back. Defaults to 7,
            max of 25.
        :param seed: Fixed random seed, useful when attempting to generate a
            specific image, defaults to None.
            May be >= 0 < 2**32.
        :return: :class:`VideoGenerateResponse` object including properties for a list
            of videos as well as a counter of total videos returned below the
            `num_videos` value due to being removed for safety.
        """
        self._validate_inputs(
            image=image,
            engine=engine,
            height=height,
            width=width,
            cfg_scale=cfg_scale,
            steps=steps,
            motion_scale=motion_scale,
            noise_aug_strength=noise_aug_strength,
            num_videos=num_videos,
            fps=fps,
            seed=seed,
        )
        if isinstance(image, Image):
            image = image.to_base64()  # type: ignore

        inputs = self._process_local_vars_to_inputs_dict(locals())

        videos = []
        endpoint = self.api_endpoint + "generate/" + engine
        output = self.infer(endpoint, inputs)
        removed_for_safety = 0
        for video_b64 in output.get("videos"):
            if video_b64.get("removed_for_safety"):
                removed_for_safety += 1
            else:
                video_b64_str = video_b64.get("video")
                video = Video(video_b64_str)  # type: ignore
                videos.append(video)
        return VideoGenerateResponse(
            videos=videos, removed_for_safety=removed_for_safety
        )

    # Purges irrelevant locals from inputs dict and converts Asset type to ids
    def _process_local_vars_to_inputs_dict(self, inputs):
        inputs.pop("self")
        for key in list(inputs):
            if inputs[key] is None:
                inputs.pop(key)
        inputs.pop("engine")
        return inputs
