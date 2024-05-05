"""Client used to infer from endpoints."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Optional

import httpx
import yaml
from pydantic import BaseModel

import octoai
from octoai import utils
from octoai.errors import OctoAIClientError, OctoAIServerError, OctoAIValidationError

LOG = logging.getLogger(__name__)

# Default timeout to account for cold starts and latency
DEFAULT_TIMEOUT_SECONDS = 900.0
DEFAULT_API_ENDPOINT = "https://api.octoai.cloud/"
SECURELINK_API_ENDPOINT = "https://api.securelink.octo.ai/"
DEFAULT_IMAGE_ENDPOINT = "https://image.octoai.run/"
SECURELINK_IMAGE_ENDPOINT = "https://image.securelink.octo.ai/"


class _JSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, bytes):
            return o.decode()
        return json.JSONEncoder.default(self, o)


class InferenceFuture(BaseModel):
    """Response class for endpoints that support server side async inferences.

    :param response_id: Unique identifier for inference
    :type response_id: str
    :param poll_url: URL to poll status of inference.
    :type poll_url: str
    """

    response_id: str
    poll_url: str


class Client:
    """A class that allows inferences from existing endpoints.

    :param token: api token, defaults to None
    :type token: str, optional
    :param config_path: path to '/.octoai/config.yaml'.  Installed in ~,
        defaults to None and will check home path
    :type config_path: str, optional
    :param timeout: seconds before request times out, defaults to 900.0
    :type timeout: float, optional
    :param verify_ssl: verify SSL certificates, defaults to True
    :type verify_ssl: bool, optional
    :param secure_link: use secure link for inferences, defaults to False
    :type secure_link: bool, optional

    Sets various headers. Gets auth token from environment if none is provided.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        config_path: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        verify_ssl: bool = True,
        secure_link: bool = False,
    ) -> None:
        """Initialize the :class: `octoai.Client` with an auth token.

        :raises :class:`OctoAIServerError`: server-side failure (unreachable, etc)
        :raises :class:`OctoAIClientError`: client-side failure (throttled, no token)
        """
        self.secure_link = secure_link

        token = token if token else os.environ.get("OCTOAI_TOKEN", None)

        if not token:
            # Default path is ~/.octoai/config.yaml for token, can be overridden
            path = Path(config_path) if config_path else Path.home()
            try:
                with open(
                    (path / Path(".octoai/config.yaml")), encoding="utf-8"
                ) as octoai_config_yaml:
                    config_dict = yaml.safe_load(octoai_config_yaml)
                token = config_dict.get("token")
            except FileNotFoundError:
                token = None

        if not token:
            logging.warning(
                "OCTOAI_TOKEN environment variable is not set. "
                + "You won't be able to reach OctoAI endpoints."
            )

        version = octoai.__version__  # type: ignore
        headers = {
            "Content-Type": "application/json",
            "user-agent": f"octoai-{version}",
        }

        if token:
            headers["Authorization"] = f"Bearer {token}"

        httpx_timeout = httpx.Timeout(timeout=timeout)
        self._httpx_client = httpx.Client(
            timeout=httpx_timeout,
            headers=headers,
            verify=verify_ssl,
        )

        if token:
            # Support calls like client.chat.completions.create()
            self.chat = octoai.chat.Chat(self)

            # Support calls like client.asset.create()
            self.asset = octoai.clients.asset_orch.AssetOrchestrator(
                token=token,
                endpoint=DEFAULT_API_ENDPOINT
                if not secure_link
                else SECURELINK_API_ENDPOINT,
            )

            # Support calls like client.tune.create()
            self.tune = octoai.clients.fine_tuning.FineTuningClient(
                token=token,
                endpoint=DEFAULT_API_ENDPOINT
                if not secure_link
                else SECURELINK_API_ENDPOINT,
            )
        else:
            logging.warning(
                "OCTOAI_TOKEN environment variable is not set. "
                "The following APIs will **not** be available: "
                "client.chat, client.asset, client.tune."
            )
            self.chat, self.asset, self.tune = None, None, None

    @staticmethod
    def _error(resp: httpx.Response):
        """Raise error of correct type for status code including message and request_id.

        :param response: httpx response

        :raises OctoAIServerError: server-side failures (unreachable, etc)
        :raises OctoAIClientError: client-side failures (throttling, unset token)
        """
        status_code = resp.status_code
        text = resp.text
        req_id_str = ""
        if resp.headers.get("X-User-Request-ID") is not None:
            req_id_str = f'(request_id: {resp.headers.get("X-User-Request-ID")})'
        text += req_id_str
        if status_code >= 500:
            raise OctoAIServerError(f"Server error: {status_code} {text}")
        elif status_code == 429:
            raise OctoAIClientError(f"Rate limit reached error: {status_code} {text}")
        else:
            raise OctoAIClientError(f"Error: {status_code} {text}")

    def infer(self, endpoint_url: str, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        """Send a request to the given endpoint URL with inputs as request body.

        :param endpoint_url: target endpoint
        :type endpoint_url: str
        :param inputs: inputs for target endpoint
        :type inputs: Mapping[str, Any]

        :raises OctoAIServerError: server-side failures (unreachable, etc)
        :raises OctoAIClientError: client-side failures (throttling, unset token)

        :return: outputs from endpoint
        :rtype: Mapping[str, Any]
        """
        resp = utils.retry(
            lambda: self._httpx_client.post(
                url=endpoint_url,
                headers={"Content-Type": "application/json"},
                content=json.dumps(inputs, cls=_JSONEncoder),
            )
        )
        if resp.status_code != 200:
            self._error(resp)
        return resp.json()

    def infer_async(
        self, endpoint_url: str, inputs: Mapping[str, Any]
    ) -> InferenceFuture:
        """Execute an inference in the background on the server.

        :class:`InferenceFuture` allows you to query status and get results
        once it's ready.

        :param endpoint_url: url to post inference request
        :type endpoint_url: str
        :param inputs: inputs to send to endpoint
        :type inputs: Mapping[str, Any]
        """
        resp = utils.retry(
            lambda: self._httpx_client.post(
                url=endpoint_url, json=inputs, headers={"X-OctoAI-Async": "1"}
            )
        )
        if resp.status_code >= 400:
            self._error(resp)
        resp_json = resp.json()
        future = InferenceFuture(**resp_json)
        return future

    def infer_stream(
        self, endpoint_url: str, inputs: Mapping[str, Any], map_fn: Callable = None
    ) -> Iterator[dict]:
        """Stream text event response body for supporting endpoints.

        This is an alternative to loading all response body into memory at once.

        :param endpoint_url: target endpoint
        :type endpoint_url: str
        :param inputs: inputs for target endpoint such as a prompt and other parameters
        :type inputs: Mapping[str, Any]
        :param inputs: function to map over each response
        :type inputs: Callable
        :return: Yields a :class:`dict` that contains the server response data.
        :rtype: Iterator[:class:`dict`]
        """
        with self._httpx_client.stream(
            method="POST",
            url=endpoint_url,
            content=json.dumps(inputs, cls=_JSONEncoder),
            headers={"accept": "text/event-stream"},
        ) as resp:
            if resp.status_code >= 400:
                # Loads response body on error for streaming responses
                resp.read()
                self._error(resp)

            for payload in resp.iter_lines():
                # Empty lines used to separate payloads.
                if payload == "":
                    continue
                # End of stream (OpenAPI /v1/chat/completions).
                elif payload == "data: [DONE]":
                    break
                # Event data identified with "data:"  JSON inside.
                elif payload.startswith("data:"):
                    payload_dict = json.loads(payload.lstrip("data:"))
                    if map_fn:
                        yield map_fn(payload_dict)
                    else:
                        yield payload_dict
                # Any other input is a malformed response.
                else:
                    raise OctoAIValidationError(
                        f"Stream response is malformed: {payload}"
                    )

    def _poll_future(self, future: InferenceFuture) -> Dict[str, str]:
        """Get from poll_url and return response.

        :param future: Future from :meth:`Client.infer_async`
        :type future: :class:`InferenceFuture`
        :raises: :class:`OctoAIClientError`
        :raises: :class:`OctoAIServerError`
        :returns: Dictionary with response
        :rtype: Dict[str, str]
        """
        response = self._httpx_client.get(url=future.poll_url)
        if response.status_code >= 400:
            self._error(response)
        return response.json()

    def is_future_ready(self, future: InferenceFuture) -> bool:
        """Return whether the future's result has been computed.

        This class will raise any errors if the status code is >= 400.

        :param future: Future from :meth:`Client.infer_async`
        :type future: :class:`InferenceFuture`
        :raises: :class:`OctoAIClientError`
        :raises: :class:`OctoAIServerError`
        :returns: True if able to use :meth:`Client.get_future_result`
        """
        resp_dict = self._poll_future(future)
        return "completed" == resp_dict.get("status")

    def get_future_result(self, future: InferenceFuture) -> Optional[Dict[str, Any]]:
        """Return the result of an inference.

        This class will raise any errors if the status code is >= 400.

        :param future: Future from :meth:`Client.infer_async`
        :type future: :class:`InferenceFuture`
        :raises: :class:`OctoAIClientError`
        :raises: :class:`OctoAIServerError`
        :returns: None if future is not ready, or dict of the response.
        :rtype: Dict[str, Any], optional
        """
        resp_dict = self._poll_future(future)
        if resp_dict.get("status") != "completed":
            return None
        response_url = resp_dict.get("response_url")
        response = self._httpx_client.get(response_url)
        if response.status_code >= 400:
            self._error(response)
        return response.json()

    def health_check(
        self, endpoint_url: str, timeout: float = DEFAULT_TIMEOUT_SECONDS
    ) -> int:
        """Check health of an endpoint using a get request.  Try until timeout.

        :param endpoint_url: URL as a str starting with https permitting get requests.
        :type endpoint_url: str
        :param timeout: Seconds before request times out, defaults to 900.
        :type timeout: float
        :return: status code from get request.  200 means ready.
        :rtype: int
        """
        resp = self._health_check(
            lambda: self._httpx_client.get(url=endpoint_url), timeout=timeout
        )
        if resp.status_code >= 400:
            self._error(resp)
        return resp.status_code

    def _health_check(
        self,
        fn: Callable[[], httpx.Response],
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        interval: float = 1.0,
        iteration_count: int = 0,
    ) -> httpx.Response:
        """Check the health of an endpoint.

        :param fn: Get request for endpoint health check.
        :type fn: Callable[[], httpx.Response]
        :param timeout: seconds before health_check times out, defaults to 900.0
        :type timeout: int, optional
        :param interval: seconds to wait before checking endpoint health again,
            defaults to 1.0
        :type interval: int, optional
        :param iteration_count: count total attempts for cold start warning,
            defaults to 0
        :type iteration_count: int
        :raises OctoAIClientError: Client-side failure such as missing api token
        :return: Response once timeout has passed
        :rtype: httpx.Response
        """
        start = time.time()
        try:
            resp = fn()
            if timeout <= 0:
                return resp
            # Raise HTTPStatusError for 4xx or 5xx.
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            if 400 <= resp.status_code < 500:
                # Raise client errors. Do not retry.
                return resp
            if iteration_count == 0 and self.__class__.__name__ == "Client":
                LOG.warning(
                    "Your endpoint may take several minutes to start and be ready to "
                    "serve inferences. You can increase your endpoint's min replicas "
                    "to mitigate cold starts."
                )
            if resp.status_code >= 500:
                stop = time.time()
                current = stop - start
                time.sleep(interval)
                return self._health_check(
                    fn, timeout - current - interval, interval, iteration_count + 1
                )
        # Raise error without retry on all other exceptions.
        return resp
