"""Generate Python examples."""

from dataclasses import dataclass
from typing import Optional, Set, cast

from octoai.documenter.common import (
    ASTVisitor,
    Buffer,
    DocumentationGenerator,
    MustacheGenerationResult,
    MustacheTemplateRegistry,
)
from octoai.documenter.json import (
    ConciseTypeFormatter,
    JSONGeneratingVisitor,
    JSONTypeFormatter,
)
from octoai.spec import SetType
from octoai.spec.ast import (
    BaseType,
    Field,
    FieldInfo,
    InferenceService,
    MediaKind,
    MediaType,
    OptionalType,
    PrimitiveType,
    PydanticHttpUrlType,
    SpecialType,
)


class PythonTypeFormatter(JSONTypeFormatter):
    """Provide a Python-like display of the types."""

    @classmethod
    def format_primitive(
        cls, primitive_type: PrimitiveType, field_info: FieldInfo
    ) -> str:
        """Show a Python-like format for the primitive type."""
        if primitive_type.type == bool:
            return "<True|False>"
        else:
            return super().format_primitive(primitive_type, field_info)

    @classmethod
    def format_set(cls, set_type: SetType, field_info: FieldInfo) -> str:
        """Show a Python-like format for the set type."""
        value = "{set of "
        value += ConciseTypeFormatter.format(set_type.member_type, field_info)
        value += "}"

        return value

    @classmethod
    def format_media(cls, media_type: MediaType, field_info: FieldInfo) -> str:
        """Show a Python-like format for the special media types."""
        if media_type.kind == MediaKind.IMAGE:
            class_name = "Image"
        elif media_type.kind == MediaKind.VIDEO:
            class_name = "Video"
        elif media_type.kind == MediaKind.AUDIO:
            class_name = "Audio"
        else:
            raise ValueError(f"unknown media type {media_type}")

        return f"{class_name}.from_url(<url string>).model_dump()"

    @classmethod
    def format_optional(cls, optional_type: OptionalType, field_info: FieldInfo) -> str:
        """Format the optional type."""
        example_value = cls._get_example_value(field_info)
        if example_value is not None:
            return example_value

        value = ConciseTypeFormatter.format(optional_type.some_type, field_info)
        value += "|None"

        return value


@dataclass
class PythonMustacheGenerationResult(MustacheGenerationResult):
    """Special generation result type to support imports."""

    discovered_special_types: Set[SpecialType]


class PythonInputGenerator(JSONGeneratingVisitor):
    """Generates a Mustache template that when rendered looks like a Python dict.

    This additionally stores all the special types encountered and return a different
    ``PythonMustacheGenerationResult``.
    """

    def __init__(
        self,
        base_buffer: Optional[Buffer] = None,
        template_registry: Optional[MustacheTemplateRegistry] = None,
    ):
        super().__init__(PythonTypeFormatter, base_buffer, template_registry)
        self._discovered_special_types: Set[SpecialType] = set()

    def generate(self, field: Field) -> MustacheGenerationResult:
        """Generate a Mustache template. This is the entry point.

        :param field: the field to generate a Mustache template for.
        :return a ``PythonMustacheGenerationResult`` containing the template, hash,
            and encountered types.
        """
        self.visit_base(field.type, field.info)
        return PythonMustacheGenerationResult(
            generated_template=self._buffer.string,
            template_hash=self._template_registry.hash,
            discovered_special_types=self._discovered_special_types,
        )

    def visit_pydantic_http_url(
        self,
        pydantic_http_url_type: PydanticHttpUrlType,
        field_info: FieldInfo,
    ):
        """Visit the Pydantic HttpUrl type."""
        self._discovered_special_types.add(pydantic_http_url_type)
        super().visit_pydantic_http_url(pydantic_http_url_type, field_info)

    def visit_media(
        self,
        media_type: MediaType,
        field_info: FieldInfo,
    ):
        """Visit the media type."""
        self._discovered_special_types.add(media_type)
        super().visit_media(media_type, field_info)


@dataclass
class PythonOutputGenerationResult:
    """Generation result type for the output portion in the Python usage example."""

    generated_string: str
    discovered_special_types: Set[SpecialType]


class PythonOutputGenerator(ASTVisitor):
    """Generates the output portion in the Python usage example."""

    def __init__(
        self,
        base_buffer: Optional[Buffer] = None,
    ):
        super().__init__()
        self._buffer = Buffer() if base_buffer is None else base_buffer
        self._discovered_special_types: Set[SpecialType] = set()

    def generate(self, field: Field) -> PythonOutputGenerationResult:
        """Generate the output usage example. This is the entry point.

        :param field: the field to generate a usage example for.
        """
        self.visit_base(field.type, field.info)
        return PythonOutputGenerationResult(
            generated_string=self._buffer.string,
            discovered_special_types=self._discovered_special_types,
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
        if isinstance(base_type, MediaType) or isinstance(base_type, SpecialType):
            super().visit_base(base_type, field_info)
            return

        self._buffer.print_string("output")

    def visit_pydantic_http_url(
        self,
        pydantic_http_url_type: PydanticHttpUrlType,
        field_info: FieldInfo,
    ):
        """Visit the Pydantic HttpUrl type.

        :param pydantic_http_url_type: the Pydantic HttpUrl type.
        :param field_info: information about the field.
        """
        self._discovered_special_types.add(pydantic_http_url_type)
        super().visit_pydantic_http_url(pydantic_http_url_type, field_info)

    def _media_kind_to_class_name(self, media_type: MediaKind) -> str:
        if media_type == MediaKind.IMAGE:
            return "Image"
        elif media_type == MediaKind.VIDEO:
            return "Video"
        elif media_type == MediaKind.AUDIO:
            return "Audio"
        else:
            raise ValueError(f"Unknown media type {media_type}")

    def visit_media(
        self,
        media_type: MediaType,
        field_info: FieldInfo,
    ):
        """Visit the media type.

        :param media_type: the media type.
        :param field_info: information about the field.
        """
        self._discovered_special_types.add(media_type)
        super().visit_media(media_type, field_info)

        self._buffer.print_string(
            f"{self._media_kind_to_class_name(media_type.kind)}"
            f".from_endpoint_response(response)"
        )


class PythonExampleGenerator(DocumentationGenerator):
    """Generate Python SDK documentation."""

    _MEDIA_KIND_TO_OCTOAI_IMPORT_NAMES = {
        MediaKind.IMAGE.value: "Image",
        MediaKind.VIDEO.value: "Video",
        MediaKind.AUDIO.value: "Audio",
    }

    def generate(self, service: InferenceService) -> MustacheGenerationResult:
        """Generate usage example given the parsed ``InferenceService``.

        :param service: the parsed ``InferenceService``.
        """
        generation_template = self._get_generation_template()

        input_result = self._generate_input(service.inference_endpoint.inputs)
        output_result = self._generate_output(service.inference_endpoint.output)
        imports_result = self._generate_imports(
            input_result.discovered_special_types
            | output_result.discovered_special_types
        )

        full_result = generation_template.format(
            input_placeholder=input_result.generated_template,
            output_placeholder=output_result.generated_string,
            imports_placeholder=imports_result,
        )

        return MustacheGenerationResult(
            generated_template=full_result,
            template_hash={
                "endpoint": "<endpoint>",
                "access_token": "<octoai_token>",
                **input_result.template_hash,
            },
        )

    def _get_generation_template(self) -> str:
        return (
            """import argparse

"""
            # Put the imports placeholder at the end. We will add a new line if
            # there is any import. (This is so that we maintain two newlines between
            # the import lines and the first line of code.
            """from octoai.client import Client{imports_placeholder}


inputs = {input_placeholder}


def main(endpoint):
    \"\"\"Run inference against the endpoint.\"\"\"
    # create an OctoAI client
    client = Client({{{{#access_token}}}}token={{{{{{access_token}}}}}}"""
            r"""{{{{/access_token}}}})

    # perform inference
    response = client.infer(endpoint_url=f"{{endpoint}}/infer", inputs=inputs)
    output = response["output"]

    print({output_placeholder})


if __name__ = "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, default="{{{{{{endpoint}}}}}}")

    args = parser.parse_args()
    main(args.endpoint)
"""
        )

    def _generate_input(self, field: Field) -> PythonMustacheGenerationResult:
        generator = PythonInputGenerator()
        result = generator.generate(field)
        return cast(PythonMustacheGenerationResult, result)

    def _generate_output(self, field: Field) -> PythonOutputGenerationResult:
        generator = PythonOutputGenerator()
        result = generator.generate(field)
        return result

    def _generate_imports(self, discovered_types: Set[SpecialType]) -> str:
        octoai_types = []

        for discovered_type in discovered_types:
            if type(discovered_type) is MediaType:
                octoai_types.append(
                    PythonExampleGenerator._MEDIA_KIND_TO_OCTOAI_IMPORT_NAMES[
                        cast(MediaType, discovered_type).kind.value
                    ]
                )

        imports = []

        if octoai_types:
            imports += [f'from octoai.types import {", ".join(octoai_types)}']

        if imports:
            return "\n" + "\n".join(imports)
        else:
            return ""
