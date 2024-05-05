"""Generate Mustache templates that, when rendered, looks like a JSON structure."""


from typing import Optional, Type, cast

from pydantic.fields import FieldInfo as PydanticFieldInfo

from octoai.documenter.common import (
    Buffer,
    MustacheGenerationResult,
    MustacheTemplateGeneratingVisitor,
    MustacheTemplateRegistry,
    TypeFormatter,
)
from octoai.spec import BaseType
from octoai.spec.ast import (
    CustomObjectType,
    DictionaryType,
    EnumType,
    Field,
    FieldInfo,
    ListType,
    MediaKind,
    MediaType,
    OptionalType,
    PrimitiveType,
    PydanticHttpUrlType,
    SetType,
    SimpleFieldInfo,
    TupleType,
    UnionType,
    UnknownType,
)


class ConciseTypeFormatter(TypeFormatter):
    """Provide a concise format of the types which shows the type names only.

    Used when the ``JSONTypeFormatter`` chooses to format container types such as
    ``List``.
    """

    @classmethod
    def format_primitive(
        cls, primitive_type: PrimitiveType, field_info: FieldInfo
    ) -> str:
        """Show the name of the primitive type."""
        if primitive_type.type == bool:
            return "<true|false>"
        if primitive_type.type == str:
            return "<string>"
        elif primitive_type.type == bytes:
            return "<base64 string>"
        elif primitive_type.type == int:
            return "<integer>"
        elif primitive_type.type == float:
            return "<number>"
        else:
            raise ValueError(f"unsupported type {primitive_type.type.__name__}")

    @classmethod
    def format_union(cls, union_type: UnionType, field_info: FieldInfo) -> str:
        """Show the name of the union type."""
        return "<union>"

    @classmethod
    def format_list(cls, list_type: ListType, field_info: FieldInfo) -> str:
        """Show the name of the list type."""
        return "<list>"

    @classmethod
    def format_tuple(cls, tuple_type: TupleType, field_info: FieldInfo) -> str:
        """Show the name of the tuple type."""
        return "<tuple>"

    @classmethod
    def format_set(cls, set_type: SetType, field_info: FieldInfo) -> str:
        """Show the name of the set type."""
        return "<set>"

    @classmethod
    def format_dictionary(cls, dict_type: DictionaryType, field_info: FieldInfo) -> str:
        """Show the name of the dictionary type."""
        return "<dict>"

    @classmethod
    def format_enum(cls, enum_type: EnumType, field_info: FieldInfo) -> str:
        """Show the name of the enum type."""
        return f"<{enum_type.name}>"

    @classmethod
    def format_unknown(cls, _unknown_type: UnknownType, field_info: FieldInfo) -> str:
        """Show the name of the unknown type."""
        return "<unknown>"

    @classmethod
    def format_optional(cls, optional_type: OptionalType, field_info: FieldInfo) -> str:
        """Show the name of the optional type."""
        return "<optional>"

    @classmethod
    def format_pydantic_http_url(
        cls, _pydantic_http_url_type: PydanticHttpUrlType, field_info: FieldInfo
    ) -> str:
        """Show the name of the pydantic HttpUrl type."""
        return "<url string>"

    @classmethod
    def format_media(cls, media_type: MediaType, field_info: FieldInfo) -> str:
        """Show the name of the media type."""
        if media_type.kind == MediaKind.IMAGE:
            return "<image>"
        elif media_type.kind == MediaKind.AUDIO:
            return "<audio>"
        elif media_type.kind == MediaKind.VIDEO:
            return "<video>"
        else:
            raise ValueError(f"unsupported media type {media_type.kind}")

    @classmethod
    def format_custom_object(
        cls, custom_object_type: CustomObjectType, field_info: FieldInfo
    ) -> str:
        """Show the name of the custom object type."""
        return "<" + custom_object_type.name + ">"


class JSONTypeFormatter(TypeFormatter):
    """Provide a JSON-like display of the types.

    If examples are provided, the first example is returned instead.
    """

    @classmethod
    def _get_example_value(cls, field_info: FieldInfo) -> Optional[str]:
        if type(field_info) is SimpleFieldInfo or (
            type(field_info) is PydanticFieldInfo and not field_info.examples
        ):
            return None

        field_info = cast(PydanticFieldInfo, field_info)
        example = field_info.examples[0]
        return str(example)

    @classmethod
    def format_primitive(
        cls, primitive_type: PrimitiveType, field_info: FieldInfo
    ) -> str:
        """Show a JSON-like format for the primitive type."""
        example_value = cls._get_example_value(field_info)
        if example_value is not None:
            if primitive_type.type == str:
                return f'"{example_value}"'
            else:
                return str(example_value)

        if primitive_type.type == bool:
            return "<true|false>"
        if primitive_type.type == str:
            return "<string>"
        elif primitive_type.type == bytes:
            return "<base64 string>"
        elif primitive_type.type == int:
            return "<integer>"
        elif primitive_type.type == float:
            return "<number>"
        else:
            raise ValueError(f"unsupported type {primitive_type.type.__name__}")

    @classmethod
    def format_union(cls, union_type: UnionType, field_info: FieldInfo) -> str:
        """Format the union type."""
        return "|".join(
            ConciseTypeFormatter.format(member, field_info)
            for member in union_type.members
        )

    @classmethod
    def format_list(cls, list_type: ListType, field_info: FieldInfo) -> str:
        """Format the list type."""
        value = "[list of "
        value += ConciseTypeFormatter.format(list_type.member_type, field_info)
        value += "]"

        return value

    @classmethod
    def format_tuple(cls, tuple_type: TupleType, field_info: FieldInfo) -> str:
        """Format the tuple type."""
        value = "["
        value += ", ".join(
            ConciseTypeFormatter.format(member, field_info)
            for member in tuple_type.members
        )
        value += "]"

        return value

    @classmethod
    def format_set(cls, set_type: SetType, field_info: FieldInfo) -> str:
        """Format the set type."""
        value = "[set of "
        value += ConciseTypeFormatter.format(set_type.member_type, field_info)
        value += "]"

        return value

    @classmethod
    def format_dictionary(cls, dict_type: DictionaryType, field_info: FieldInfo) -> str:
        """Format the dictionary type."""
        value = "{"
        value += ConciseTypeFormatter.format(dict_type.key_type, field_info)
        value += ": "
        value += ConciseTypeFormatter.format(dict_type.value_type, field_info)
        value += "}"

        return value

    @classmethod
    def format_enum(cls, enum_type: EnumType, field_info: FieldInfo) -> str:
        """Format the enum type."""
        example_value = cls._get_example_value(field_info)
        if example_value is not None:
            return example_value

        formatted_variants = []
        for _, variant_value in enum_type.variants:
            if type(variant_value) is str:
                formatted_variants += [f'"{variant_value}"']
            else:
                formatted_variants += [str(variant_value)]
        return "|".join(formatted_variants)

    @classmethod
    def format_unknown(cls, _unknown_type: UnknownType, field_info: FieldInfo) -> str:
        """Format the unknown type."""
        return "<unknown>"

    @classmethod
    def format_optional(cls, optional_type: OptionalType, field_info: FieldInfo) -> str:
        """Format the optional type."""
        example_value = cls._get_example_value(field_info)
        if example_value is not None:
            return example_value

        value = ConciseTypeFormatter.format(optional_type.some_type, field_info)
        value += "|null"

        return value

    @classmethod
    def format_pydantic_http_url(
        cls, _pydantic_http_url_type: PydanticHttpUrlType, field_info: FieldInfo
    ) -> str:
        """Format the Pydantic HttpUrl type."""
        example_value = cls._get_example_value(field_info)
        if example_value:
            return f'"{example_value}"'

        return "<url string>"

    @classmethod
    def format_media(cls, media_type: MediaType, field_info: FieldInfo) -> str:
        """Format the media type."""
        example_value = cls._get_example_value(field_info)
        if example_value is not None:
            return example_value

        return '{"data": {"url": "<url string>"}}'

    @classmethod
    def format_custom_object(
        cls, custom_object_type: CustomObjectType, field_info: FieldInfo
    ) -> str:
        """Format the custom object type."""
        example_value = cls._get_example_value(field_info)
        if example_value is not None:
            return example_value

        return "<" + custom_object_type.name + ">"


class JSONGeneratingVisitor(MustacheTemplateGeneratingVisitor):
    """Generates a Mustache template that when rendered looks like a JSON structure."""

    _MAX_TRAVERSAL_DEPTH = 5

    def __init__(
        self,
        type_formatter_class: Type[TypeFormatter] = JSONTypeFormatter,
        base_buffer: Optional[Buffer] = None,
        template_registry: Optional[MustacheTemplateRegistry] = None,
    ):
        self._type_formatter_class = type_formatter_class
        super().__init__(base_buffer, template_registry)

    def generate(self, field: Field) -> MustacheGenerationResult:
        """Generate a Mustache template. This is the entry point.

        :param field: the field to generate a Mustache template for.
        """
        self.visit_base(field.type, field.info)
        return MustacheGenerationResult(
            generated_template=self._buffer.string,
            template_hash=self._template_registry.hash,
        )

    def visit_base(
        self,
        base_type: BaseType,
        field_info: FieldInfo,
    ):
        """Visit the base type.

        :param base_type: the base type.
        :param field_info: information about the field.
        """
        if type(base_type) is CustomObjectType:
            self.visit_custom_object(cast(CustomObjectType, base_type), field_info)
            return

        super().visit_base(base_type, field_info)

        value = self._type_formatter_class.format(base_type, field_info)
        self._template_registry.register_example(self._field_traversal_stack, value)
        self._buffer.print_string(
            self._template_registry.get_field_template_tag(self._field_traversal_stack)
        )

    def visit_custom_object(
        self,
        custom_object_type: CustomObjectType,
        field_info: FieldInfo,
    ):
        """Visit the custom object type.

        :param custom_object_type: The custom object type.
        :param field_info: information about the field.
        """
        # Add a space here to prevent colliding with Mustache opening brackets. This is
        # because we don't start a new line immediately when we see a field to avoid
        # display strangeness in the rendered string.
        self._buffer.print_string("{ ")

        # Filter out fields that will not be shown. This is needed to figure out when we
        # want to hide the commas. Note that this is purely to work around Mustache’s
        # logic-less limitations; we may consider switching to other methods in the
        # future.
        displayed_fields = {
            field_name: field_item
            for field_name, field_item in custom_object_type.fields.items()
            if self._should_display_field(field_item.info)
        }

        for idx, (field_name, field_item) in enumerate(displayed_fields.items()):
            self._field_traversal_stack.append(field_name)
            if len(self._field_traversal_stack) >= type(self)._MAX_TRAVERSAL_DEPTH:
                raise ValueError("Deeply nested objects are currently unsupported")

            if type(field_item.type) is not CustomObjectType:
                self._buffer.print_string(
                    self._template_registry.get_field_opening_format_tag(
                        self._field_traversal_stack
                    )
                )

            self._buffer.increase_indent()
            self._buffer.new_line()

            self._buffer.print_string('"' + field_name + '"' + ": ")

            self.visit_base(field_item.type, field_item.info)
            if idx < len(displayed_fields) - 1:
                self._buffer.print_string(",")

            if type(field_item.type) is not CustomObjectType:
                self._buffer.print_string(
                    self._template_registry.get_field_closing_format_tag(
                        self._field_traversal_stack
                    )
                )

            self._field_traversal_stack.pop()
            self._buffer.decrease_indent()

        self._buffer.new_line()
        self._buffer.print_string("}")
