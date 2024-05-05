"""OctoAI Image Generation."""
from __future__ import (
    annotations,  # required to allow 3.7+ python use type | syntax introduced in 3.10
)

from enum import Enum
from typing import Dict, List

from clients.image_gen.models import ImageEncoding, Scheduler, SDXLStyles
from clients.image_gen.models.pre_defined_styles import PreDefinedStyles
from octoai.client import Client
from octoai.clients.asset_orch import Asset
from octoai.errors import OctoAIValidationError
from octoai.types import Image

SDXL_ALLOWABLE_WIDTH_TO_HEIGHT = {
    1024: {1024, 768},
    896: {1152, 896},
    832: {1216, 512},
    768: {1344, 1024, 576, 512},
    704: {1216, 384},
    640: {1536, 768, 640},
    576: {1024, 448, 768},
    512: {832, 768, 704, 512},
    448: {576},
    384: {704},
    1536: {640},
    1344: {768},
    1216: {832, 704},
    1152: {896},
}

SD_ALLOWABLE_WIDTH_TO_HEIGHT = {
    576: {1024, 768},
    512: {512, 704, 768},
    640: {512, 768},
    768: {512},
    1024: {576},
}

SSD_ALLOWABLE_WIDTH_TO_HEIGHT = {
    640: {1536},
    768: {1344},
    832: {1216},
    896: {1152},
    1024: {1024},
    1152: {896},
    1216: {832},
    1344: {768},
    1536: {640},
}

MAX_CFG_SCALE = 50
MAX_NUM_IMAGES = 16
MAX_STEPS = 200


class Engine(str, Enum):
    """
    SDXL: Stable Diffusion XL
    SD: Stable Diffusion
    """

    SDXL = "sdxl"
    SD = "sd"
    SSD = "ssd"
    CONTROLNET_SDXL = "controlnet-sdxl"
    CONTROLNET_SD = "controlnet-sd"

    def to_name(self):
        if self == self.SDXL:
            return "Stable Diffusion XL"
        elif self == self.SD:
            return "Stable Diffusion"
        elif self == self.SSD:
            return "Stable Diffusion SSD"
        elif self == self.CONTROLNET_SDXL:
            return "ControlNet Stable Diffusion XL"
        elif self == self.CONTROLNET_SD:
            return "ControlNet Stable Diffusion 1.5"


class ImageGenerateResponse:
    """
    Image generation response.

    Contains a list of images as well as a counter of those filtered for safety.
    """

    def __init__(self, images: List[Image], removed_for_safety: int):
        self._images = images
        self._removed_for_safety = removed_for_safety

    @property
    def images(self) -> List[Image]:
        """Return list of :class:`Image` generated from request."""
        return self._images

    @property
    def removed_for_safety(self) -> int:
        """Return int representing number of images removed for safety."""
        return self._removed_for_safety


class ImageGenerator(Client):
    """Client for image generation."""

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
        super(ImageGenerator, self).__init__(*args, **kwargs)

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
        engine: Engine | str,
        cfg_scale: float | None,
        height: int | None,
        high_noise_frac: float | None,
        num_images: int | None,
        seed: int | None,
        steps: int | None,
        strength: float | None,
        width: int | None,
        image_encoding: ImageEncoding | str | None,
        sampler: Scheduler | str | None,
        prompt_2: str | None,
        negative_prompt_2: str | None,
        use_refiner: bool | None,
        init_image: str | None = None,
        controlnet: str | None = None,
        controlnet_image: str | None = None,
        controlnet_conditioning_scale: float | None = None,
        style_preset: str | None = None,
    ):
        """Validate inputs."""
        engines = [e.value for e in Engine]
        if engine not in engines:
            raise OctoAIValidationError(
                f"engine set to {engine}.  Must be one of: {', '.join(engines)}."
            )

        # Check only compatible with engine attributes being used
        if high_noise_frac and engine == "sd":
            self._input_not_match_engine(
                "high_noise_frac", high_noise_frac, engine, "sdxl"
            )
        if prompt_2 and engine == "sd":
            self._input_not_match_engine("prompt_2", prompt_2, engine, "sdxl")
        if negative_prompt_2 and engine == "sd":
            self._input_not_match_engine(
                "negative_prompt_2", negative_prompt_2, engine, "sdxl"
            )
        if use_refiner is not None and engine == "sd":
            self._input_not_match_engine("use_refiner", use_refiner, engine, "sdxl")

        # Check number values in range
        if cfg_scale is not None and ((0 > cfg_scale) or (MAX_CFG_SCALE < cfg_scale)):
            raise OctoAIValidationError(
                f"cfg_scale set to: {cfg_scale}. Allowable range is > 0 and <= {MAX_CFG_SCALE}."
            )

        if high_noise_frac is not None and (0 > high_noise_frac or high_noise_frac > 1):
            raise OctoAIValidationError(
                f"high_noise_frac set to: {high_noise_frac}. Allowable range is "
                f">= 0 and <= 1."
            )
        if num_images is not None and (0 >= num_images or num_images > MAX_NUM_IMAGES):
            raise OctoAIValidationError(
                f"num_images set to: {num_images}.  Allowable range is > 0 and <= {MAX_NUM_IMAGES}."
            )
        if isinstance(seed, list):
            for each in seed:
                if each is not None and (0 > each or each >= 2**32):
                    raise OctoAIValidationError(
                        f"seed({seed}) contains {each}. Allowable range is >= 0 and "
                        f"< 2**32."
                    )
        if type(seed) == int and (0 > seed or seed >= 2**32):
            raise OctoAIValidationError(
                f"seed set to: {seed}.  Allowable range is >= 0 and < 2**32."
            )
        if steps is not None and (0 >= steps or steps > MAX_STEPS):
            raise OctoAIValidationError(
                f"steps set to: {steps}.  Allowable range is > 0 and <= {MAX_STEPS}."
            )
        if strength is not None and (0 > strength or strength > 1):
            raise OctoAIValidationError(
                f"strength set to: {strength}.  Allowable range is >= 0 and <= 1."
            )
        if (
            controlnet_conditioning_scale is not None
            and controlnet_conditioning_scale < 0
        ):
            raise OctoAIValidationError(
                f"controlnet_conditional_scale set to: {controlnet_conditioning_scale}."
                " Allowable range is >= 0."
            )

        # More verifying required value
        if strength is not None and init_image is None:
            raise OctoAIValidationError(
                f"init_image required for img2img generation.  "
                f"strength({strength}) cannot be set if "
                f"init_image is None."
            )

        if style_preset is not None and engine == "sd":
            self._input_not_match_engine("style_preset", style_preset, engine, "sdxl")
        if style_preset is not None and engine == "sdxl":
            try:
                style_preset = PreDefinedStyles(style_preset)
            except ValueError:
                msg = (
                    f"style_preset({style_preset}) is not valid.  "
                    f"Valid options include {[e.value for e in SDXLStyles]}."
                )
                raise OctoAIValidationError(msg)

        # Validate width and height to engine
        if (height is None) ^ (width is None):
            raise OctoAIValidationError(
                f"if height({height}) or width({width}) is set "
                f"to None, both must be None."
            )
        engine = Engine(engine)
        if height is not None and width is not None:
            self._validate_height_and_width_to_engine(engine, height, width)

        if init_image is not None:
            if not isinstance(init_image, Image):
                init_image = Image(init_image)
            if not init_image.is_valid():
                msg = (
                    "init_image is not a valid image.  May either use the "
                    "octoai.types Image class or a base64 string."
                )
                raise OctoAIValidationError(msg)

        if controlnet_image is not None:
            if not isinstance(controlnet_image, Image):
                controlnet_image = Image(controlnet_image)
            if not controlnet_image.is_valid():
                msg = (
                    "controlnet_image is not a valid image.  May either use the "
                    "octoai.types Image class or a base64 string."
                )
                raise OctoAIValidationError(msg)

        if engine in [
            Engine.CONTROLNET_SDXL,
            Engine.CONTROLNET_SDXL.value,
            Engine.CONTROLNET_SD,
            Engine.CONTROLNET_SD.value,
        ]:
            if controlnet_image is None:
                raise OctoAIValidationError(
                    f"controlnet_image is required for engine {engine}."
                )

            if controlnet is None:
                raise OctoAIValidationError(
                    f"controlnet is required for engine {engine}."
                )

    # Server will return a 500 error if incorrect height and width are entered
    @staticmethod
    def _validate_height_and_width_to_engine(engine: Engine, height: int, width: int):
        width_to_height_by_engine = {
            Engine.SDXL.value: SDXL_ALLOWABLE_WIDTH_TO_HEIGHT,
            Engine.SD.value: SD_ALLOWABLE_WIDTH_TO_HEIGHT,
            Engine.SSD.value: SSD_ALLOWABLE_WIDTH_TO_HEIGHT,
            Engine.CONTROLNET_SDXL.value: SDXL_ALLOWABLE_WIDTH_TO_HEIGHT,
            Engine.CONTROLNET_SD.value: SD_ALLOWABLE_WIDTH_TO_HEIGHT,
        }

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

    # May be worthwhile to limit steps to 50, but at this time, 100 matches the UI
    # num_images limit of 10 temp set by Itay, but part of larger limit
    # discussion.
    def generate(
        self,
        engine: Engine | str,
        prompt: str,
        prompt_2: str | None = None,  # SDXL only
        negative_prompt: str | None = None,
        negative_prompt_2: str | None = None,  # SDXL only
        checkpoint: str | Asset | None = None,
        vae: str | Asset | None = None,
        textual_inversions: Dict[str | Asset, str] | None = None,
        loras: Dict[str | Asset, float] | None = None,
        sampler: str | Scheduler | None = None,  # Server default DDIM
        height: int | None = None,  # Different defaults for sdxl and sd
        width: int | None = None,
        cfg_scale: float | None = 12.0,
        steps: int | None = 30,
        num_images: int | None = 1,
        seed: int | List[int] | None = None,
        init_image: str | Image | None = None,  # b64, img2img only
        controlnet: str | None = None,  # controlnet-sdxl|sd only
        controlnet_image: str | Image | None = None,  # b64, controlnet-sdxl|sd only
        controlnet_conditioning_scale: float
        | None = None,  # 1.0 server default, controlnet-sdxl|sd only
        strength: float | None = None,  # 0.8 server default, img2img only
        style_preset: str | SDXLStyles | None = None,
        use_refiner: bool | None = None,  # True default, SDXL only
        high_noise_frac: float | None = None,  # 0.8 server default, SDXL only
        enable_safety: bool | None = True,
        image_encoding: ImageEncoding | str = None,
    ) -> ImageGenerateResponse:
        """
        Generate a list of images based on request.

        :param engine: Required. "sdxl" for Stable Diffusion XL; "sd" for Stable
            Diffusion 1.5; "ssd" for Stable Diffusion SSD; "controlnet-sdxl" for
            ControlNet Stable Diffusion XL; "controlnet-sd" for ControlNet Stable
            Diffusion 1.5.
        :param prompt: Required. Describes the image to generate.
            ex. "An octopus playing chess, masterpiece, photorealistic"
        :param prompt_2: High level description of the image to generate,
            defaults to None.
        :param negative_prompt: Description of image traits to avoid, defaults to None.
            ex. "Fingers, distortions"
        :param negative_prompt_2: High level description of things to avoid during
            generation, defaults to None.
            ex. "Unusual proportions and distorted faces"
        :param checkpoint: Which checkpoint to use for inferences, defaults to None.
        :param vae: Custom VAE to be used during image generation, defaults to None.
        :param textual_inversions: A dictionary of textual inversion updates,
            defaults to None
            ex. {'name': 'trigger_word'}
        :param loras: A dictionary of LoRAs updates to apply and their weight, can also
            be used with Assets created in the SDK directly, defaults to None.
            ex. {'crayon-style': 0.3, my_created_asset: 0.1}
        :param sampler: :class:`Scheduler` to use when generating image,
            defaults to None.
        :param height: Height of image to generate, defaults to None.
        :param width: Width of image to generate, defaults to None.
        :param cfg_scale: How closely to adhere to prompt description, defaults to 12.0.
            Must be >= 0 and <= 50.
        :param steps: How many steps of diffusion to run, defaults to 30.
            May be > 0 and <= 100.
        :param num_images: How many images to generate, defaults to 1.
            May be > 0 and <= 4.
        :param seed: Fixed random seed, useful when attempting to generate a
            specific image, defaults to None.
            May be >= 0 < 2**32.
        :param init_image: Starting image for img2img mode, defaults to None.
            Requires a b64 string image or :class:`Image`.
        :param controlnet: String matching id of controlnet to use for controlnet engine
            inferences, defaults to None.  Required for using controlnet engines.
        :param controlnet_image: Starting image for controlnet-sdxl mode, defaults to None.
            Requires a b64 string image or :class:`Image`.
        :param controlnet_conditioning_scale: How strong the effect of the controlnet
            should be, defaults to 1.0.
        :param strength: How much creative to be in img2img mode, defaults to 0.8.
            May be >= 0 and <= 1.  Must have an `init_image`.
        :param style_preset: Used to guide the output image towards a particular style,
            only usable with SDXL,defaults to None.
            ex. "low-poly"
        :param use_refiner: Whether to apply the sdxl refiner, defaults to True.
        :param high_noise_frac: Which fraction of steps to perform with the base model,
            defaults to 0.8.
            May be >= 0 and <= 1.
        :param enable_safety: Whether to use safety checking on generated outputs or
            not, defaults to True.
        :param image_encoding: Choose returned :class:`ImageEncoding` type,
            defaults to :class:`ImageEncoding.JPEG`.
        :return: :class:`GenerateImagesResponse` object including properties for a list
            of images as well as a counter of total images returned below the
            `num_images` value due to being removed for safety.
        """
        self._validate_inputs(
            engine,
            cfg_scale,
            height,
            high_noise_frac,
            num_images,
            seed,
            steps,
            strength,
            width,
            image_encoding,
            sampler,
            prompt_2,
            negative_prompt_2,
            use_refiner,
            init_image,
            controlnet,
            controlnet_image,
            controlnet_conditioning_scale,
            style_preset,
        )
        if isinstance(init_image, Image):
            init_image = init_image.to_base64()
        if isinstance(controlnet_image, Image):
            controlnet_image = controlnet_image.to_base64()

        inputs = self._process_local_vars_to_inputs_dict(locals())

        images = []
        endpoint = self.api_endpoint + "generate/" + engine
        output = self.infer(endpoint, inputs)
        removed_for_safety = 0
        for image_b64 in output.get("images"):
            if image_b64.get("removed_for_safety"):
                removed_for_safety += 1
            else:
                image_b64_str = image_b64.get("image_b64")
                image = Image(image_b64_str)
                images.append(image)
        return ImageGenerateResponse(
            images=images, removed_for_safety=removed_for_safety
        )

    # If the key is of type str, it does nothing, otherwise it returns a dict where
    # objects with id fields have the id field used as the key instead.
    # Examples are Assets as keys.
    @staticmethod
    def _replace_object_keys_with_ids(obj_dict: dict) -> dict:
        result = {}
        for key, value in obj_dict.items():
            if isinstance(key, Asset):
                key = key.id
            elif type(key) != str:
                msg = (
                    f"key({key}) is invalid.  `loras` and `textual_inversions` "
                    f"require keys in dictionary to either be a str "
                    f"or :class:`Asset`."
                )
                raise OctoAIValidationError(msg)
            result[key] = value
        return result

    # Purges irrelevant locals from inputs dict and converts Asset type to ids
    def _process_local_vars_to_inputs_dict(self, inputs):
        inputs.pop("self")
        for key in list(inputs):
            if inputs[key] is None:
                inputs.pop(key)
        inputs.pop("engine")
        if "loras" in inputs:
            inputs["loras"] = self._replace_object_keys_with_ids(inputs["loras"])
        if "textual_inversions" in inputs:
            inputs["textual_inversions"] = self._replace_object_keys_with_ids(
                inputs["textual_inversions"]
            )
        if isinstance(inputs.get("vae"), Asset):
            inputs["vae"] = inputs["vae"].id
        if isinstance(inputs.get("checkpoint"), Asset):
            inputs["checkpoint"] = inputs["checkpoint"].id
        return inputs
