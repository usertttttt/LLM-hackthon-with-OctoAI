"""
Contains the base interface that OctoAI endpoints should implement.

Developers that want to create an endpoint should implement the
``Service`` class in this module as directed by the ``octoai`` command-line
interface.
"""
import functools
import inspect
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Mapping, Tuple, Type

import pydantic_core
from fastapi import Form
from pydantic import BaseModel, Field, create_model

from .types import File

DEFAULT_VOLUME_PATH = "/octoai/cache"
STORE_ASSETS_NOT_OVERRIDDEN = "NOT_OVERRIDDEN"


def volume_path() -> str:
    """Get mounted volume path in docker.

    :return: Docker path.
    """
    docker_path = os.environ.get("OCTOAI_VOLUME_PATH", None)
    if docker_path:
        return docker_path
    else:
        return DEFAULT_VOLUME_PATH


VOLUME_ENVIRONMENT = {
    "HUGGINGFACE_HUB_CACHE": os.path.join(volume_path(), "huggingface_hub_cache"),
    "TORCH_HOME": os.path.join(volume_path(), "torch_home"),
}


class Service(ABC):
    """
    The base interface that OctoAI endpoints should implement.

    Developers that want to create an endpoint should implement this
    class as directed by the ``octoai`` command-line interface.
    """

    def setup(self) -> None:
        """
        Perform service initialization.

        A common operation to include here is loading weights and making
        those available to the ``infer()`` method in a member variable.
        """
        pass

    def store_assets(self) -> None:
        """Download model assets."""
        pass

    def on_server_startup(self) -> None:
        """
        Perform any necessary initialization when the server starts.

        This method is separate from setup() because setup() can be called
        outside the serving context to include weights when building the image.
        """
        pass

    def on_server_shutdown(self) -> None:
        """Perform any necessary cleanup when the server stops."""
        pass

    @abstractmethod
    def infer(self, **kwargs: Any) -> Any:
        """Perform inference."""
        pass

    setattr(store_assets, STORE_ASSETS_NOT_OVERRIDDEN, True)


class ResponseAnalytics(BaseModel):
    """Additional analytics metadata."""

    inference_time_ms: float = Field(
        description="Inference execution time (without pauses)"
    )
    performance_time_ms: float = Field(
        description="Inference execution time (including pauses)"
    )


def inspect_input_types(method: Callable) -> Type[BaseModel]:
    """Create Pydantic input model from inference method signature."""
    signature = inspect.signature(method)

    if len(signature.parameters) < 1:
        raise ValueError(f"{method.__name__}() requires at least one argument")

    return inspect_input_parameters(signature.parameters)


def inspect_input_parameters(
    parameters: Mapping[str, inspect.Parameter]
) -> Type[BaseModel]:
    """Create Pydnatic input mode from the provided signature."""
    args = OrderedDict()

    for arg_name, arg in parameters.items():
        if arg.annotation == inspect.Parameter.empty:
            raise ValueError("infer() requires type annotations for args")

        default = (
            pydantic_core.PydanticUndefined
            if arg.default == inspect.Parameter.empty
            else arg.default
        )
        args[arg_name] = (
            arg.annotation,
            Field(default=default),
        )

    return create_model(
        "Input",
        __config__=None,
        __base__=BaseModel,
        __module__=__name__,
        __validators__=None,
        **args,
    )


def inspect_output_types(method: Callable) -> Type[BaseModel]:
    """Create Pydantic output model from inference method signature."""
    signature = inspect.signature(method)

    if signature.return_annotation == inspect._empty:
        raise ValueError(f"{method.__name__}() requires a return type annotation")

    rets = OrderedDict()
    rets["output"] = (signature.return_annotation, None)
    rets["analytics"] = (ResponseAnalytics, None)

    return create_model(
        "Output",
        __config__=None,
        __base__=BaseModel,
        __module__=__name__,
        __validators__=None,
        **rets,
    )


def implements_form_data(service: Service) -> bool:
    """Check if ``infer_form_data()`` is implemented."""
    for name, method in inspect.getmembers(service, predicate=inspect.ismethod):
        if name == "infer_form_data":
            return True

    return False


def transform_form_data_signature(service: Service) -> inspect.Signature:
    """Add FastAPI Form() defaults to 'str' fields and check compatibility."""
    if not hasattr(service.infer_form_data, "__path__"):  # type: ignore[attr-defined]
        raise ValueError(
            "infer_form_data() must be decorated with @path() "
            "when using form-data inference."
        )

    if service.infer_form_data.__path__ == "/infer":  # type: ignore[attr-defined]
        raise ValueError("Cannot use /infer for form-data inference.")

    params = []
    signature = inspect.signature(service.infer_form_data)  # type: ignore[attr-defined]
    for param in signature.parameters.values():
        if param.annotation == File:
            params.append(param)
        elif param.annotation == str:
            params.append(param.replace(default=Form()))
        else:
            raise ValueError("Only File and str params are supported for form data.")

    if not params:
        raise ValueError("infer_form_data() must have at least one parameter.")

    return inspect.Signature(params)


def find_additional_endpoints(service: Service) -> Mapping[str, Tuple[Callable, str]]:
    """Find additional methods that should be exposed as endpoints."""
    methods = {}
    reserved_methods = [attr for attr in dir(Service) if not attr.startswith("_")]
    reserved_methods.append("infer_form_data")

    for name, method in inspect.getmembers(service, predicate=inspect.ismethod):
        if name not in reserved_methods and not name.startswith("_"):
            methods[name] = (method, getattr(method, "__path__", None))

    return methods


def path(path: str):
    """Specify the path for a service method."""

    def wrapped(fn: Callable):
        @functools.wraps(fn)
        def wrapped_fn(*args, **kwargs):
            return fn(*args, **kwargs)

        wrapped_fn.__path__ = path  # type: ignore[attr-defined]
        return wrapped_fn

    return wrapped
