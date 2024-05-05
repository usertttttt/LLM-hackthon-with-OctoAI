"""
Entities and APIs for working with text generation models.

Instead of using these classes directly, developers should
use the octoai.client.Client class. For example:

client = octoai.client.Client()
completion = client.chat.completions.create(...)
"""

from enum import Enum
from typing import Iterable, List, Optional, Union

from pydantic import BaseModel, ValidationError
from typing_extensions import Literal

from clients.ollm.models.chat_completion_response_format import (
    ChatCompletionResponseFormat,
)
from clients.ollm.models.chat_message import ChatMessage
from clients.ollm.models.create_chat_completion_request import (
    CreateChatCompletionRequest,
)
from octoai.client import Client
from octoai.errors import OctoAIValidationError

TEXT_DEFAULT_ENDPOINT = "https://text.octoai.run/v1/chat/completions"
TEXT_SECURELINK_ENDPOINT = "https://text.securelink.octo.ai/v1/chat/completions"


class TextModel(str, Enum):
    """List of available text models."""

    LLAMA_2_13B_CHAT = "llama-2-13b-chat"
    LLAMA_2_70B_CHAT = "llama-2-70b-chat"
    CODELLAMA_7B_INSTRUCT = "codellama-7b-instruct"
    CODELLAMA_13B_INSTRUCT = "codellama-13b-instruct"
    CODELLAMA_34B_INSTRUCT = "codellama-34b-instruct"
    CODELLAMA_70B_INSTRUCT = "codellama-70b-instruct"
    MISTRAL_7B_INSTRUCT = "mistral-7b-instruct"
    MIXTRAL_8X7B_INSTRUCT = "mixtral-8x7b-instruct"

    def to_name(self):
        """Return the name of the model."""
        if self == self.LLAMA_2_13B_CHAT:
            return "llama-2-13b-chat"
        elif self == self.LLAMA_2_70B_CHAT:
            return "llama-2-70b-chat"
        elif self == self.CODELLAMA_7B_INSTRUCT:
            return "codellama-7b-instruct"
        elif self == self.CODELLAMA_13B_INSTRUCT:
            return "codellama-13b-instruct"
        elif self == self.CODELLAMA_34B_INSTRUCT:
            return "codellama-34b-instruct"
        elif self == self.CODELLAMA_70B_INSTRUCT:
            return "codellama-70b-instruct"
        elif self == self.MISTRAL_7B_INSTRUCT:
            return "mistral-7b-instruct"
        elif self == self.MIXTRAL_8X7B_INSTRUCT:
            return "mixtral-8x7b-instruct"


def get_model_list() -> List[str]:
    """Return a list of available text models."""
    return [model.value for model in TextModel]


class ChoiceDelta(BaseModel):
    """Contents for streaming text completion responses."""

    content: Optional[str] = None
    role: Optional[Literal["system", "user", "assistant", "tool"]] = None


class Choice(BaseModel):
    """A single choice in a text completion response."""

    index: int
    message: ChatMessage = None
    delta: ChoiceDelta = None
    finish_reason: Optional[
        Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
    ] = None


class CompletionUsage(BaseModel):
    """Usage statistics for a text completion response."""

    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ChatCompletion(BaseModel):
    """A text completion response."""

    id: str
    choices: List[Choice]
    created: int
    model: str
    object: Optional[Literal["chat.completion", "chat.completion.chunk"]] = None
    system_fingerprint: Optional[str] = None
    usage: Optional[CompletionUsage] = None


class Completions:
    """Text completions API."""

    client: Client
    endpoint: str = TEXT_DEFAULT_ENDPOINT

    def __init__(self, client: Client) -> None:
        self.client = client

        if self.client.secure_link:
            self.endpoint = TEXT_SECURELINK_ENDPOINT

    def create(
        self,
        *,
        messages: List[ChatMessage],
        model: Union[str, TextModel],
        frequency_penalty: Optional[float] = 0.0,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = 0.0,
        response_format: Optional[ChatCompletionResponseFormat] = None,
        stop: Optional[str] = None,
        stream: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 1.0,
    ) -> Union[ChatCompletion, Iterable[ChatCompletion]]:
        """
        Create a chat completion with a text generation model.

        :param messages: Required. A list of messages to use as context for the
            completion.
        :param model: Required. The model to use for the completion. Supported models
            are listed in the `octoai.chat.TextModel` enum.
        :param frequency_penalty: Positive values make it less likely that the model
            repeats tokens several times in the completion. Valid values are between
            -2.0 and 2.0.
        :param max_tokens: The maximum number of tokens to generate.
        :param presence_penalty: Positive values make it less likely that the model
            repeats tokens in the completion. Valid values are between -2.0 and 2.0.
        :param response_format: An object specifying the format that the model must
            output.
        :param stop: A list of sequences where the model stops generating tokens.
        :param stream: Whether to return a generator that yields partial message
            deltas as they become available, instead of waiting to return the entire
            response.
        :param temperature: Sampling temperature. A value between 0 and 2. Higher values
            make the model more creative by sampling less likely tokens.
        :param top_p: The cumulative probability of the most likely tokens to use. Use
            `temperature` or `top_p` but not both.
        """
        request = CreateChatCompletionRequest(
            messages=messages,
            model=model.value if isinstance(model, TextModel) else model,
            frequency_penalty=frequency_penalty,
            function_call=None,
            functions=None,
            logit_bias=None,
            max_tokens=max_tokens,
            n=1,
            presence_penalty=presence_penalty,
            response_format=response_format,
            stop=stop,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
            user=None,
        )

        inputs = request.to_dict()

        if stream:
            return self.client.infer_stream(
                self.endpoint, inputs, map_fn=lambda resp: ChatCompletion(**resp)
            )  # type: ignore

        resp = self.client.infer(self.endpoint, inputs)
        try:
            return ChatCompletion(**resp)
        except ValidationError as e:
            raise OctoAIValidationError(
                "Unable to validate response from server.", caused_by=e
            )


class Chat:
    """Chat API for text generation models."""

    completions: Completions

    def __init__(self, client: Client):
        self.completions = Completions(client)
