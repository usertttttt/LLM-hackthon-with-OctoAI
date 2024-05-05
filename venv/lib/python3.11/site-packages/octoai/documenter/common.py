"""Base classes for generation of template."""

from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, cast

from pydantic.fields import FieldInfo as PydanticFieldInfo
from pydantic_core import PydanticUndefined

from octoai.spec.ast import (
    BaseType,
    CustomObjectType,
    DictionaryType,
    EnumType,
    Field,
    FieldInfo,
    InferenceService,
    ListType,
    MediaType,
    OptionalType,
    PrimitiveType,
    PydanticHttpUrlType,
    SetType,
    SimpleFieldInfo,
    SpecialType,
    TupleType,
    UnionType,
    UnknownType,
)


class Buffer:
    """Holds a string of the generated template and manages the indentation level."""

    _INDENTATION_LEVEL = 4

    def __init__(self, indentation_level: int = 0):
        self._generated_strings = ""
        self._indentation_level = indentation_level

    def increase_indent(self):
        """Increase the indent by one level."""
        self._indentation_level += 1

    def decrease_indent(self):
        """Decrease the indent by one level."""
        self._indentation_level -= 1

    def print_line(self, line: str, indented: bool = True):
        """Append the line to the end of the buffer.

        :param line: the line to be added
        :param indented: whether to add indentation to the start of the line
        """
        self._generated_strings += self._get_indentation_string() + line
        self.new_line(indented=indented)

    def print_string(self, s: str):
        """Append the string to the end of the buffer.

        :param s: the string to be added
        """
        self._generated_strings += s

    def new_line(self, indented: bool = True):
        """Add a new line.

        :param indented: whether to add indentation to the start of the line
        """
        self._generated_strings += "\n"
        if indented:
            self._generated_strings += self._get_indentation_string()

    def _get_indentation_string(self) -> str:
        return " " * (Buffer._INDENTATION_LEVEL * self._indentation_level)

    @property
    def string(self) -> str:
        """Return the contents."""
        return str(self)

    def __str__(self):
        """Return the contents."""
        return self._generated_strings


TemplateRegistryHashType = Union[str, Dict[str, "TemplateRegistryHashType"]]


class MustacheTemplateRegistry:
    """Store for all the Mustache tags and hashes."""

    def __init__(
        self,
        base_hash: Dict[str, TemplateRegistryHashType] = None,
        field_prefix: str = "",
    ):
        self._hash = base_hash if base_hash else {}
        self._field_prefix = field_prefix

    def register_example(self, key_path: List[str], value: str):
        """Register a new value at the key path.

        :param key_path: a key path referencing the object.
        :param value: the value for the object in the key path.
        """
        path = copy(key_path)

        # Iteratively update or assign the data.
        current_dict = self._hash
        for idx, key in enumerate(path):
            if idx == len(key_path) - 1:
                current_dict[key] = value
                return

            if key not in current_dict:
                current_dict[key] = {}
            current_dict = cast(Dict[str, TemplateRegistryHashType], current_dict[key])

    @property
    def hash(self) -> Dict[str, TemplateRegistryHashType]:
        """Return the hash."""
        return self._hash

    def _get_field_name(self, key_path: List[str]) -> str:
        return self._field_prefix + ".".join(key_path)

    def get_field_template_tag(self, key_path: List[str]) -> str:
        """Get the template tag for the object at key path.

        Triple braces are used to disable Mustache’s escaping for HTML symbols like
        '<' and '>'.

        :param key_path: the referenced key path.
        """
        return "{{{" + self._get_field_name(key_path) + "}}}"

    def get_field_opening_format_tag(self, key_path: List[str]) -> str:
        """Get the opening tag for the object at key path.

        :param key_path: the referenced key path.
        """
        return "{{#" + self._get_field_name(key_path) + "}}"

    def get_field_closing_format_tag(self, key_path: List[str]) -> str:
        """Get the closing tag for the object at key path.

        :param key_path: the referenced key path.
        """
        return "{{/" + self._get_field_name(key_path) + "}}"


@dataclass
class MustacheGenerationResult:
    """Generated Mustache template string and the hashes."""

    generated_template: str
    template_hash: Dict[str, TemplateRegistryHashType]


class ASTVisitor(ABC):
    """Visitor for visiting all the AST nodes."""

    def visit_base(
        self,
        base_type: BaseType,
        field_info: FieldInfo,
    ):
        """Visit the base type.

        :param base_type: the base type.
        :param field_info: information about the field.
        """
        if isinstance(base_type, SpecialType):
            self.visit_special(cast(SpecialType, base_type), field_info)
        elif type(base_type) is CustomObjectType:
            self.visit_custom_object(cast(CustomObjectType, base_type), field_info)
        elif type(base_type) is PrimitiveType:
            self.visit_primitive(cast(PrimitiveType, base_type), field_info)
        elif type(base_type) is UnionType:
            self.visit_union(cast(UnionType, base_type), field_info)
        elif type(base_type) is ListType:
            self.visit_list(cast(ListType, base_type), field_info)
        elif type(base_type) is TupleType:
            self.visit_tuple(cast(TupleType, base_type), field_info)
        elif type(base_type) is SetType:
            self.visit_set(cast(SetType, base_type), field_info)
        elif type(base_type) is DictionaryType:
            self.visit_dict(cast(DictionaryType, base_type), field_info)
        elif isinstance(base_type, EnumType):
            self.visit_enum(cast(EnumType, base_type), field_info)
        elif type(base_type) is UnknownType:
            self.visit_unknown(cast(UnknownType, base_type), field_info)
        elif type(base_type) is OptionalType:
            self.visit_optional(cast(OptionalType, base_type), field_info)
        else:
            raise NotImplementedError(
                f"the type '{base_type}' is currently unsupported"
            )

    def visit_special(
        self,
        special_type: SpecialType,
        field_info: FieldInfo,
    ):
        """Visit the special type.

        :param special_type: the special type.
        :param field_info: information about the field.
        """
        if type(special_type) is MediaType:
            self.visit_media(cast(MediaType, special_type), field_info)
        elif type(special_type) is PydanticHttpUrlType:
            self.visit_pydantic_http_url(
                cast(PydanticHttpUrlType, special_type), field_info
            )
        else:
            raise NotImplementedError(
                f"the type '{type(special_type).__name__}' is currently unsupported"
            )

    def visit_primitive(
        self,
        primitive_type: PrimitiveType,
        field_info: FieldInfo,
    ):
        """Visit the primitive type.

        :param primitive_type: the primitive type.
        :param field_info: information about the field.
        """

    def visit_union(self, union_type: UnionType, field_info: FieldInfo):
        """Visit the union type.

        :param union_type: the union type.
        :param field_info: information about the field.
        """

    def visit_list(
        self,
        list_type: ListType,
        field_info: FieldInfo,
    ):
        """Visit the list type.

        :param list_type: the list type.
        :param field_info: information about the field.
        """
        return

    def visit_tuple(
        self,
        tuple_type: TupleType,
        field_info: FieldInfo,
    ):
        """Visit the tuple type.

        :param tuple_type: the tuple type.
        :param field_info: information about the field.
        """

    def visit_set(
        self,
        set_type: SetType,
        field_info: FieldInfo,
    ):
        """Visit the set type.

        :param set_type: the set type.
        :param field_info: information about the field.
        """

    def visit_dict(
        self,
        dict_type: DictionaryType,
        field_info: FieldInfo,
    ):
        """Visit the dict type.

        :param dict_type: the dict type.
        :param field_info: information about the field.
        """

    def visit_enum(
        self,
        enum_type: EnumType,
        field_info: FieldInfo,
    ):
        """Visit the enum type.

        :param enum_type: the enum type.
        :param field_info: information about the field.
        """

    def visit_unknown(self, unknown_type: UnknownType, field_info: FieldInfo):
        """Visit the unknown type.

        :param unknown_type: the unknown type.
        :param field_info: information about the field.
        """

    def visit_optional(self, optional_type: OptionalType, field_info: FieldInfo):
        """Visit the optional type.

        :param optional_type: the optional type.
        :param field_info: information about the field.
        """

    def visit_media(
        self,
        media_type: MediaType,
        field_info: FieldInfo,
    ):
        """Visit the media type.

        :param media_type: the media type.
        :param field_info: information about the field.
        """

    def visit_pydantic_http_url(
        self,
        pydantic_http_url_type: PydanticHttpUrlType,
        field_info: FieldInfo,
    ):
        """Visit the Pydantic HttpUrl type.

        :param pydantic_http_url_type: the Pydantic HttpUrl type.
        :param field_info: information about the field.
        """

    def visit_custom_object(
        self,
        custom_object_type: CustomObjectType,
        field_info: FieldInfo,
    ):
        """Visit the custom object type.

        :param custom_object_type: The custom object type.
        :param field_info: information about the field.
        """


class MustacheTemplateGeneratingVisitor(ASTVisitor, ABC):
    """A subclass of ``ASTVisitor`` that stores states for Mustache templates."""

    def __init__(
        self,
        base_buffer: Optional[Buffer] = None,
        template_registry: Optional[MustacheTemplateRegistry] = None,
    ):
        super().__init__()
        self._buffer = base_buffer if base_buffer is not None else Buffer()
        self._template_registry = (
            template_registry
            if template_registry is not None
            else MustacheTemplateRegistry()
        )
        self._field_traversal_stack: List[str] = []

    def _should_display_field(self, field_info: FieldInfo) -> bool:
        if isinstance(field_info, SimpleFieldInfo):
            return field_info.default is None
        elif isinstance(field_info, PydanticFieldInfo):
            return cast(PydanticFieldInfo, field_info).default == PydanticUndefined

    @abstractmethod
    def generate(self, field: Field) -> MustacheGenerationResult:
        """Generate the Mustache template for the ``Field``.

        :param field: the ``Field`` to generate a Mustache template for.
        """
        raise NotImplementedError

    def visit_base(
        self,
        base_type: BaseType,
        field_info: FieldInfo,
    ):
        """Visit the base type, only if a default value is not provided for the field.

        :param base_type: the base type.
        :param field_info: information about the field.
        """
        if not self._should_display_field(field_info):
            return

        super().visit_base(base_type, field_info)


class DocumentationGenerator(ABC):
    """Base class of all documentation generators."""

    @abstractmethod
    def generate(self, service: InferenceService) -> MustacheGenerationResult:
        """Generate documentation for the inference service.

        :param service: the inference service.
        """
        raise NotImplementedError


class TypeFormatter(ABC):
    """Format types for display."""

    @classmethod
    def format(cls, base_type: BaseType, field_info: FieldInfo) -> str:
        """Format the base type. This is a base function for ``BaseType``s.

        :param base_type: the base type.
        :param field_info: information about the field.
        """
        if type(base_type) is PrimitiveType:
            return cls.format_primitive(cast(PrimitiveType, base_type), field_info)
        elif type(base_type) is UnknownType:
            return cls.format_unknown(cast(UnknownType, base_type), field_info)
        elif isinstance(base_type, SpecialType):
            return cls.format_special(cast(SpecialType, base_type), field_info)
        elif type(base_type) is CustomObjectType:
            return cls.format_custom_object(
                cast(CustomObjectType, base_type), field_info
            )
        elif type(base_type) is UnionType:
            return cls.format_union(cast(UnionType, base_type), field_info)
        elif type(base_type) is ListType:
            return cls.format_list(cast(ListType, base_type), field_info)
        elif type(base_type) is TupleType:
            return cls.format_tuple(cast(TupleType, base_type), field_info)
        elif type(base_type) is SetType:
            return cls.format_set(cast(SetType, base_type), field_info)
        elif type(base_type) is DictionaryType:
            return cls.format_dictionary(cast(DictionaryType, base_type), field_info)
        elif type(base_type) is EnumType:
            return cls.format_enum(cast(EnumType, base_type), field_info)
        elif type(base_type) is OptionalType:
            return cls.format_optional(cast(OptionalType, base_type), field_info)
        else:
            return "<type>"

    @classmethod
    def format_special(cls, special_type: SpecialType, field_info: FieldInfo) -> str:
        """Format the special type. This is a base function for ``SpecialType``s.

        :param special_type: the special type.
        :param field_info: information about the field.
        """
        if type(special_type) is PydanticHttpUrlType:
            return cls.format_pydantic_http_url(
                cast(PydanticHttpUrlType, special_type), field_info
            )
        elif type(special_type) is MediaType:
            return cls.format_media(cast(MediaType, special_type), field_info)
        else:
            return "<special>"

    @classmethod
    @abstractmethod
    def format_primitive(
        cls, primitive_type: PrimitiveType, field_info: FieldInfo
    ) -> str:
        """Format the primitive type.

        :param primitive_type: the primitive type.
        :param field_info: information about the field.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def format_union(cls, union_type: UnionType, field_info: FieldInfo) -> str:
        """Format the union type.

        :param union_type: the primitive type.
        :param field_info: information about the field.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def format_list(cls, list_type: ListType, field_info: FieldInfo) -> str:
        """Format the list type.

        :param list_type: the list type.
        :param field_info: information about the field.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def format_tuple(cls, tuple_type: TupleType, field_info: FieldInfo) -> str:
        """Format the tuple type.

        :param tuple_type: the tuple type.
        :param field_info: information about the field.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def format_set(cls, set_type: SetType, field_info: FieldInfo) -> str:
        """Format the set type.

        :param set_type: the set type.
        :param field_info: information about the field.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def format_dictionary(cls, dict_type: DictionaryType, field_info: FieldInfo) -> str:
        """Format the dictionary type.

        :param dict_type: the dictionary type.
        :param field_info: information about the field.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def format_enum(cls, enum_type: EnumType, field_info: FieldInfo) -> str:
        """Format the enum type.

        :param enum_type: the enum type.
        :param field_info: information about the field.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def format_unknown(cls, unknown_type: UnknownType, field_info: FieldInfo) -> str:
        """Format the unknown type.

        :param unknown_type: the unknown type.
        :param field_info: information about the field.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def format_optional(cls, optional_type: OptionalType, field_info: FieldInfo) -> str:
        """Format the optional type.

        :param optional_type: the optional type.
        :param field_info: information about the field.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def format_pydantic_http_url(
        cls, pydantic_http_url_type: PydanticHttpUrlType, field_info: FieldInfo
    ) -> str:
        """Format the pydantic HttpUrl type.

        :param pydantic_http_url_type: the pydantic HttpUrl type.
        :param field_info: information about the field.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def format_media(cls, media_type: MediaType, field_info: FieldInfo) -> str:
        """Format the media type.

        :param media_type: the media type.
        :param field_info: information about the field.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def format_custom_object(
        cls, custom_object_type: CustomObjectType, field_info: FieldInfo
    ) -> str:
        """Format the custom object type.

        :param custom_object_type: the custom object type.
        :param field_info: information about the field.
        """
        raise NotImplementedError
