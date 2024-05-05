"""Nodes of the parsed AST for the ``octoai.Service`` subclass."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Type, Union

from pydantic.fields import FieldInfo as PydanticFieldInfo


@dataclass(frozen=True)
class BaseType:
    """Describes a type."""

    pass


@dataclass(frozen=True)
class SpecialType(BaseType):
    """Describes a special type is specially handled, e.g., ``octoai.types.Image``."""

    pass


FieldInfo = Union["SimpleFieldInfo", PydanticFieldInfo]
"""Either a ``SimpleFieldInfo`` or a ``pydantic.fields.FieldInfo"""


@dataclass(frozen=True)
class Field:
    """Describes a field in an object."""

    type: BaseType
    info: FieldInfo


@dataclass(frozen=True)
class NamedField(Field):
    """Describes a named field in an object."""

    name: str


@dataclass(frozen=True)
class SimpleFieldInfo:
    """Contains the info for non-Pydantic fields (e.g., method signature)."""

    default: Optional[str]
    description: Optional[str]


@dataclass(frozen=True)
class PrimitiveType(BaseType):
    """Describes primitive types."""

    type: Union[
        Type[str],
        Type[bool],
        Type[bytes],
        Type[int],
        Type[float],
    ]


@dataclass(frozen=True)
class UnionType(BaseType):
    """Describes types annotated with ``typing.Union``."""

    members: List[BaseType]


@dataclass(frozen=True)
class ListType(BaseType):
    """Describes types annotated with ``typing.List``."""

    member_type: BaseType


@dataclass(frozen=True)
class TupleType(BaseType):
    """Describes types annotated with ``typing.Tuple``."""

    members: List[BaseType]


@dataclass(frozen=True)
class SetType(BaseType):
    """Describes types annotated with ``typing.Set``."""

    member_type: BaseType


@dataclass(frozen=True)
class DictionaryType(BaseType):
    """Describes types annotated with ``typing.Dict``."""

    key_type: BaseType
    value_type: BaseType


@dataclass(frozen=True)
class EnumType(BaseType):
    """Describes enum types."""

    name: str
    variants: List[Tuple[str, Union[int, str]]]


@dataclass(frozen=True)
class UnknownType(BaseType):
    """
    Description for types that are not fully specified.

    Examples: ``typing.List`` without type arguments.
    """

    pass


@dataclass(frozen=True)
class OptionalType(BaseType):
    """Describes types annotated with ``typing.Optional``."""

    some_type: BaseType


@dataclass(frozen=True)
class CustomObjectType(BaseType):
    """Describes custom Pydantic ``BaseModel`` subclasses defined by the user."""

    name: str
    fields: Dict[str, Field]


class MediaKind(Enum):
    """Describes the type of the special ``MediaType``."""

    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"


@dataclass(frozen=True)
class MediaType(SpecialType):
    """Describes a special media type such as ``octoai.types.Image``."""

    kind: MediaKind


@dataclass(frozen=True)
class PydanticHttpUrlType(SpecialType):
    """Describes the HttpUrl type provided by Pydantic."""

    pass


@dataclass(frozen=True)
class InferenceEndpoint:
    """Describes the signature of the inference endpoint."""

    inputs: Field
    output: Field


@dataclass(frozen=True)
class InferenceService:
    """Describes the inference service."""

    inference_endpoint: InferenceEndpoint
