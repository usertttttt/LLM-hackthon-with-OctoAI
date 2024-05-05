"""An AST for the parsed ``octoai.service.Service`` subclass."""

import enum
import functools
import inspect
import sys
import typing
from inspect import Signature
from typing import Any, Callable, Dict, List, Set, Tuple, Type, Union

import typing_extensions
from pydantic import BaseModel, HttpUrl
from pydantic.fields import FieldInfo as PydanticFieldInfo
from pydantic_core import Url

from octoai.service import Service, inspect_input_parameters
from octoai.spec.ast import (
    BaseType,
    CustomObjectType,
    DictionaryType,
    EnumType,
    Field,
    FieldInfo,
    InferenceEndpoint,
    InferenceService,
    ListType,
    MediaKind,
    MediaType,
    NamedField,
    OptionalType,
    PrimitiveType,
    PydanticHttpUrlType,
    SetType,
    SimpleFieldInfo,
    TupleType,
    UnionType,
    UnknownType,
)
from octoai.types import Audio, Image, Video


def parse_service(service_class: Type[Service]) -> InferenceService:
    """Parse the ``Service`` subclass and return the AST of the parsed service."""
    # Make sure this is a subclass of `Service` but not `Service` itself.
    if not issubclass(service_class, Service) or service_class == Service:
        raise ValueError(
            "The provided service module must be a subclass of octoai.Service"
        )

    inference_endpoint = _parse_inference_endpoint(service_class.infer)
    return InferenceService(inference_endpoint=inference_endpoint)


def _parse_inference_endpoint(infer: Callable[..., Any]) -> InferenceEndpoint:
    """Introspect the input/output from the ``infer()`` signature."""
    signature = inspect.signature(infer)
    service_input_parameters = signature.parameters
    # The conditional statement here uses < 2 as the condition because the `Service`
    # subclass must have at least one parameter. Because we are looking at the
    # class-level type, `self` is not bound and thus there is an extra parameter.
    if len(service_input_parameters) < 2:
        raise ValueError("infer() requires type annotations for args")

    # Remove `self`, but now we need to get the parameters again.
    infer = functools.partial(infer, None)
    parameters = inspect.signature(infer).parameters
    parsed_input = _parse_field(inspect_input_parameters(parameters))

    service_output = signature.return_annotation
    if service_output == Signature.empty:
        raise ValueError("infer() requires return type annotation")
    parsed_output = _parse_field(service_output)

    return InferenceEndpoint(inputs=parsed_input, output=parsed_output)


def _parse_field(return_annotation: Any) -> Field:
    """Parse a field in an object."""
    return _parse_type_annotation(return_annotation)


def _parse_type_annotation(annotation: Any) -> Field:
    parameter_type_annotation = type(annotation)

    if typing.get_origin(parameter_type_annotation) is typing_extensions.Annotated or (
        sys.version_info >= (3, 9)
        and typing.get_origin(parameter_type_annotation) is typing.Annotated
    ):
        return _parse_annotated_type_annotation(annotation)
    else:
        return _parse_unit_type_annotation(annotation)


def _parse_unit_type_annotation(annotation: Any) -> Field:
    return Field(
        type=parse_type(annotation),
        info=SimpleFieldInfo(
            default=None,
            description=None,
        ),
    )


# No annotation for the Annotation type because Python 3.8 does not have ``Annotated``.
def _parse_annotated_type_annotation(annotation) -> Field:  # type: ignore[valid-type] # noqa
    # FastAPI requires the first annotation argument to be the actual type. We follow
    # the same requirement here.
    args = typing.get_args(annotation)

    if len(args) < 1:
        raise ValueError(
            "annotated types on the signature in infer() must have the base type"
        )

    base_type = args[0]

    # Search the remaining annotations for Pydantic Field.
    info: FieldInfo = SimpleFieldInfo(default=None, description=None)
    for arg in args[1:]:
        if type(arg) is PydanticFieldInfo:
            info = arg

    return Field(
        type=parse_type(base_type),
        info=info,
    )


def parse_type(data_type: Type[Any]) -> BaseType:
    """Parse the annotated type and return a subclass of ``BaseType`` in the AST."""
    if (
        data_type == str
        or data_type == bytes
        or data_type == int
        or data_type == float
        or data_type == bool
    ):
        return PrimitiveType(type=data_type)
    elif data_type == Video:
        return MediaType(kind=MediaKind.VIDEO)
    elif data_type == Audio:
        return MediaType(kind=MediaKind.AUDIO)
    elif data_type == Image:
        return MediaType(kind=MediaKind.IMAGE)
    elif data_type == HttpUrl or data_type == Url:
        return PydanticHttpUrlType()
    elif inspect.isclass(data_type) and issubclass(data_type, enum.Enum):
        return _parse_enum_type(data_type)
    elif inspect.isclass(data_type) and issubclass(data_type, BaseModel):
        return _parse_custom_object_type(data_type)
    elif type(data_type) is typing._GenericAlias and (  # type: ignore[attr-defined] # noqa
        typing.get_origin(data_type) == list
        or typing.get_origin(data_type) == tuple
        or typing.get_origin(data_type) == dict
        or typing.get_origin(data_type) == set
    ):
        return _parse_aggregation_type(data_type)
    elif (
        isinstance(data_type, typing._GenericAlias)  # type: ignore[attr-defined] # noqa
        and typing.get_origin(data_type) == Union
    ):
        return _parse_union_type(data_type)
    else:
        raise NotImplementedError(
            f"currently the `{data_type.__name__}` type is unsupported"
        )


def _parse_enum_type(enum_type: Type[enum.Enum]):
    variants_and_representations = [
        (member, variant.value) for (member, variant) in enum_type.__members__.items()
    ]
    return EnumType(name=enum_type.__name__, variants=variants_and_representations)


def _parse_aggregation_type(
    aggregation_type: Union[Type[Dict], Type[List], Type[Tuple], Type[Set]]
) -> BaseType:
    args = typing.get_args(aggregation_type)

    if aggregation_type.__origin__ == list:  # type: ignore[union-attr] # noqa
        return ListType(
            member_type=parse_type(args[0]) if len(args) == 1 else UnknownType()
        )
    elif aggregation_type.__origin__ == set:  # type: ignore[union-attr] # noqa
        return SetType(
            member_type=parse_type(args[0]) if len(args) == 1 else UnknownType()
        )
    elif aggregation_type.__origin__ == tuple:  # type: ignore[union-attr] # noqa
        members = [parse_type(arg) for arg in args]
        return TupleType(members=members if members else [UnknownType()])
    elif aggregation_type.__origin__ == dict:  # type: ignore[union-attr] # noqa
        key_type: BaseType = UnknownType()
        value_type: BaseType = UnknownType()

        if len(args) > 0:
            key_type = parse_type(args[0])
            value_type = parse_type(
                args[1]
            )  # Python already checks if ValueType is specified.

        return DictionaryType(key_type=key_type, value_type=value_type)
    else:
        raise ValueError(
            f"the `{aggregation_type.__name__}` is not an aggregation type"
        )


def _parse_union_type(union_type: Type[Union[Any]]) -> BaseType:
    args = typing.get_args(union_type)
    if len(args) == 2 and args[1] is type(None):
        return OptionalType(some_type=parse_type(args[0]))
    return UnionType(members=[parse_type(arg) for arg in args])


def _parse_pydantic_field_info(name: str, field_info: PydanticFieldInfo) -> Field:
    # Pydantic’s ``FieldInfo`` provides a trove of information. Just attach it.
    return NamedField(
        name=name,
        type=parse_type(field_info.annotation),
        info=field_info,
    )


def _parse_custom_object_type(custom_object_type: Type[BaseModel]) -> BaseType:
    model_fields = custom_object_type.model_fields

    fields = {}
    for field_name, field in model_fields.items():
        parsed_field = _parse_pydantic_field_info(field_name, field)
        fields[field_name] = parsed_field

    return CustomObjectType(name=custom_object_type.__name__, fields=fields)
