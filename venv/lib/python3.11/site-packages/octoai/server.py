"""
Server for OctoAI endpoints created with the ``octoai`` CLI.

Developers that want to create an endpoint should not use
this module directly. Instead, they should use the ``octoai``
command-line interface, which directs them to implement the
``octoai.service.Service`` class and use the ``octoai`` CLI to help
build and deploy their endpoint.
"""
import asyncio
import dataclasses
import enum
import importlib
import inspect
import json
import logging
import multiprocessing
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from http import HTTPStatus
from multiprocessing import Pipe, Process, Queue
from multiprocessing.connection import Connection
from typing import Any, Dict, NamedTuple, Optional, Type

import chevron
import click
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from octoai import documenter
from octoai.documenter import MustacheGenerationResult

from .service import (
    STORE_ASSETS_NOT_OVERRIDDEN,
    VOLUME_ENVIRONMENT,
    ResponseAnalytics,
    Service,
    find_additional_endpoints,
    implements_form_data,
    inspect_input_types,
    inspect_output_types,
    transform_form_data_signature,
)

_LOG = logging.getLogger(__name__)

_OCTOAI_SERVICE_MODULE = "octoai.service"
_OCTOAI_BASE_SERVICE_CLASS = "Service"

_PREDICT_LOOP_WATCHDOG_SECONDS = 2
"""Delay in seconds between checking if the predict loop is running."""

_process_mutex = multiprocessing.Lock()
"""Lock for spawning predict loop."""

_TERMINATE_PREDICT_LOOP_REQUEST = "terminate"
"""Sent to predict loop process to terminate it."""


_LOAD_MODEL_MESSAGE = "load_model"
"""Sent from predict loop when it is loading the model."""


_MODEL_LOADED_MESSAGE = "model_loaded"
"""Sent from predict loop when it is ready for inference."""

_STORE_ASSET_NEEDED_CODE = 10
"""Exit status code when store asset is required."""

_STORE_ASSET_NOT_NEEDED_CODE = 11
"""Exit status code when store asset is not required."""


class ServiceMethod(enum.Enum):
    """Class for distinguishing route implementations in the request queue."""

    INFER_JSON = "infer"
    INFER_FORM_DATA = "infer_form_data"


class InferenceRequest(NamedTuple):
    """Class for returning inference results."""

    response_pipe: Connection
    method: str
    inputs: Any


class InferenceResponse(NamedTuple):
    """Class for returning inference results."""

    inference_time_ms: float
    outputs: Any


def is_store_assets_needed(ctx) -> bool:
    """Check if store-assets step is required."""

    def check_store_assets_overriden(queue, ctx, dummy) -> None:
        service = load_service(
            ctx.parent.params["service_module"],
            class_name=ctx.parent.params["service_class"],
        )
        queue.put(not hasattr(service.store_assets, STORE_ASSETS_NOT_OVERRIDDEN))

    queue: Queue[bool] = Queue()
    p = Process(target=check_store_assets_overriden, args=(queue, ctx, 0))
    p.start()
    p.join(timeout=10)
    result = queue.get()
    return result


def maybe_set_volume_environment_variables(ctx) -> None:
    """Set volume environment variables if needed."""
    if not is_store_assets_needed(ctx):
        return
    for key, value in VOLUME_ENVIRONMENT.items():
        os.environ[key] = value


def _predict_loop(service, _request_queue):
    """Loop which handles prediction requests.

    This loop runs for the duration of the server and receives prediction
    requests posted to the _REQUEST_QUEUE. When the request is done processing
    the results are posted to the response_pipe where they are handled by
    the main /predict endpoint.
    """
    startup_pipe = _request_queue.get()
    startup_pipe.send(_LOAD_MODEL_MESSAGE)
    try:
        service.setup()
    except Exception as e:
        _LOG.error("_predict_loop: model setup failed with {e}", exc_info=1)
        startup_pipe.send(e)
        startup_pipe.close()
        return

    startup_pipe.send(_MODEL_LOADED_MESSAGE)
    startup_pipe.close()

    def signal_handler(_signum, _frame):
        # This will only kill the _predict_loop process, not the parent
        sys.exit()

    signal.signal(signal.SIGINT, signal_handler)

    while True:
        try:
            inference_request = _request_queue.get()
            if (
                isinstance(inference_request, type(_TERMINATE_PREDICT_LOOP_REQUEST))
                and inference_request == _TERMINATE_PREDICT_LOOP_REQUEST
            ):
                sys.exit()

            try:
                start_time = time.perf_counter_ns()
                infer_fn = getattr(service, inference_request.method)
                results = infer_fn(**inference_request.inputs)
                stop_time = time.perf_counter_ns()
                response = InferenceResponse((stop_time - start_time) / 1e9, results)
            except Exception as e:
                _LOG.error("infer() raised Exception", exc_info=1)
                response = e
            inference_request.response_pipe.send(response)
            inference_request.response_pipe.close()
        except Exception:
            # We only end up here if something went wrong outside the predict call
            # continue loop
            pass


class Server:
    """
    Server for OctoAI endpoints created with the ``octoai`` CLI.

    Developers that want to create an endpoint should not use
    this class directly. Instead, they should use the ``octoai``
    command-line interface, which directs them to implement the
    ``octoai.service.Service`` class and use the ``octoai`` CLI to
    help build and deploy their endpoint.
    """

    class State(enum.Enum):
        """Describes the states of Server."""

        UNINITIALIZED = "UNINITIALIZED"
        LAUNCH_PREDICT_LOOP = "LAUNCH_PREDICT_LOOP"
        SETUP_SERVICE = "SETUP_SERVICE"
        RUNNING = "RUNNING"
        SHUTTING_DOWN = "SHUTTING_DOWN"
        STOPPED = "STOPPED"

    def __init__(self, service: Service, async_enable: bool = True):
        self.app = FastAPI(lifespan=lambda _: self.prepare_for_serving())
        self.service: Service = service
        self._state = self.State.UNINITIALIZED
        self.is_async = async_enable
        self._request_queue: multiprocessing.Queue[Any] = None
        self._predict_loop_watchdog_task: asyncio.Task = None

        self.response_headers = {
            "OCTOAI_REPLICA_NAME": os.environ.get("OCTOAI_REPLICA_NAME", ""),
        }

        # Build Pydantic models for input/ouput for /infer route
        Input = inspect_input_types(service.infer)
        Output = inspect_output_types(service.infer)

        # Build Pydantic models for input/output for additional routes
        Inputs, Outputs = {}, {}
        additional_endpoints = find_additional_endpoints(service)
        for method_name, method_info in additional_endpoints.items():
            method, _ = method_info
            Inputs[method_name] = inspect_input_types(method)
            Outputs[method_name] = inspect_output_types(method)

        # Used to read from the request queue in async mode
        async def _pipe_reader(read: Connection):
            """Async multiprocessing.Pipe reader.

            :param read: pipe file handle to read from.
            :return: the contents of the pipe when read.
            """
            data_available = asyncio.Event()
            asyncio.get_event_loop().add_reader(read.fileno(), data_available.set)
            if not read.poll():
                await data_available.wait()
            result = read.recv()
            data_available.clear()
            asyncio.get_event_loop().remove_reader(read.fileno())
            return result

        # Async mode: Put request in the queue for predict loop
        # Sync mode: Call predict method directly
        async def _infer_common(
            service_method: str,
            method_args: Dict[str, Any],
            output_type: Type[BaseModel],
        ) -> Response:
            if not self.is_running:
                return JSONResponse(
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                    content={"status": self.state.name},
                )

            if self.is_async:
                read_conn, write_conn = Pipe()
                start_perf = time.perf_counter_ns()
                request = InferenceRequest(write_conn, service_method, method_args)
                self._request_queue.put(request)
                response = await _pipe_reader(read_conn)
                performance_time_ms = (time.perf_counter_ns() - start_perf) / 1e6
                if isinstance(response, Exception):
                    raise response

                prediction = response.outputs
                inference_time_ms = response.inference_time_ms
            else:
                # track time elapsed in nanoseconds only while app is not asleep
                start_process = time.process_time_ns()
                # track time elapsed in nanoseconds including any sleep time
                start_perf = time.perf_counter_ns()
                infer_fn = getattr(service, service_method)
                prediction = infer_fn(**method_args)
                inference_time_ms = (time.process_time_ns() - start_process) / 1e6
                performance_time_ms = (time.perf_counter_ns() - start_perf) / 1e6

            return Response(
                status_code=HTTPStatus.OK,
                headers=self.response_headers,
                media_type="application/json",
                content=output_type(
                    output=prediction,
                    analytics=ResponseAnalytics(
                        inference_time_ms=inference_time_ms,
                        performance_time_ms=performance_time_ms,
                    ),
                ).model_dump_json(),
            )

        # Implementation for form data route.
        # This function signature is dynamically redefined later based on that of
        # Service.infer_form_data() if it is implemented, so that FastAPI
        # can know the parameters and their types when registering the route.
        async def infer_form_data(**kwargs):
            return await _infer_common(
                service_method=ServiceMethod.INFER_FORM_DATA.value,
                method_args=kwargs,
                output_type=OutputFormData,
            )

        # Add form data route to FastAPI if implemented.
        if implements_form_data(service):
            infer_form_data.__signature__ = transform_form_data_signature(service)
            OutputFormData = inspect_output_types(service.infer_form_data)
            _LOG.info(
                "adding endpoint infer_form_data() as %s",
                service.infer_form_data.__path__,
            )
            self.app.add_api_route(
                path=service.infer_form_data.__path__,
                endpoint=infer_form_data,
                methods=["POST"],
                response_model=OutputFormData,
            )

        @self.app.get("/healthcheck")
        def health() -> JSONResponse:
            return JSONResponse(
                status_code=HTTPStatus.OK
                if self.is_running
                else HTTPStatus.SERVICE_UNAVAILABLE,
                content={"status": self.state.name, "async_enable": self.is_async},
            )

        @self.app.get("/")
        def root() -> JSONResponse:
            return JSONResponse(
                status_code=HTTPStatus.OK,
                content={
                    "docs": "/docs",
                    "openapi": "/openapi.json",
                },
            )

        @self.app.post(
            "/infer",
            response_model=Output,
        )
        async def infer(request: Input) -> Response:
            return await _infer_common(
                service_method=ServiceMethod.INFER_JSON.value,
                method_args={k: v for k, v in request},
                output_type=Output,
            )

        # Create endpoint function for additional endpoints
        def _get_infer_endpoint_fn(method_name: str):
            async def _infer_endpoint(request: Inputs[method_name]) -> Response:
                return await _infer_common(
                    service_method=method_name,
                    method_args={k: v for k, v in request},
                    output_type=Outputs[method_name],
                )

            return _infer_endpoint

        # Add additional endpoints to FastAPI
        for method_name, method_info in additional_endpoints.items():
            method, method_path = method_info
            method_path = method_path or f"/{method_name}".replace("_", "-")
            _LOG.info("adding endpoint %s() as %s", method_name, method_path)
            self.app.add_api_route(
                path=method_path,
                endpoint=_get_infer_endpoint_fn(method_name),
                methods=["POST"],
                response_model=Outputs[method_name],
            )

    @property
    def is_running(self):
        """True when this server instance can serve a request."""
        return self.state == self.State.RUNNING

    @property
    def state(self):
        """Get the status of this server."""
        return self._state

    @state.setter
    def state(self, new_state):
        """Set the status of the server, and log transition."""
        _LOG.info("status: %s -> %s", self._state, new_state)
        self._state = new_state

    @asynccontextmanager
    async def prepare_for_serving(self):
        """Context manager that should surround all serving.

        This is intended to be used as an ASGI application's lifetime handler.
        """
        assert self.state in (
            self.State.UNINITIALIZED,
            self.State.STOPPED,
        ), f"prepare_for_serving: status not UNINITIALIZED or STOPPED: {self.state}"

        _LOG.info("lifecycle: on_server_startup")
        self.service.on_server_startup()

        if self.is_async:
            self.state = self.State.LAUNCH_PREDICT_LOOP
            # load_into_memory is handled in predict loop so that subprocess
            # does all GPU access.
            self._start_predict_loop()
        else:
            self.state = self.State.SETUP_SERVICE
            self.service.setup()
            self.state = self.State.RUNNING

        yield

        self.state = self.State.SHUTTING_DOWN

        if self.is_async:
            self._stop_predict_loop()

        _LOG.info("lifecycle: on_server_shutdown")
        self.service.on_server_shutdown()
        self.state = self.State.STOPPED

    async def _check_predict_loop(self):
        if not self._predict_process.is_alive():
            self._start_predict_loop()

    def _start_predict_loop(self):
        context = multiprocessing.get_context("spawn")
        if not self._request_queue:
            # Only need to create this queue once. This function may be called
            # multiple times if the predict loop dies.
            self._request_queue = context.Queue()

        self._predict_process = context.Process(
            target=_predict_loop,
            name="_predict_loop",
            args=(self.service, self._request_queue),
        )
        self._predict_process.start()

        read_conn, write_conn = Pipe()

        def _remove_reader():
            asyncio.get_event_loop().remove_reader(read_conn)
            read_conn.close()

        def read_startup_message():
            try:
                value = read_conn.recv()
            except EOFError:
                asyncio.get_event_loop().remove_reader(read_conn)
                return

            if isinstance(value, Exception):
                _LOG.error("predict loop died during setup", exc_info=value)
                # Though unconventional, this appears the be the proper way to cleanly
                # terminate uvicorn. It's not easy to get ahold of
                # uvicorn.Server.handle_exit() from here.
                self._stop_predict_loop()
                os.kill(os.getpid(), signal.SIGTERM)

            if value == _LOAD_MODEL_MESSAGE:
                if self.state != self.State.LAUNCH_PREDICT_LOOP:
                    _LOG.error(
                        "_read_predict_loop_startup_message: %s message: "
                        "not in expected state: %s",
                        value,
                        self.state,
                    )
                    _remove_reader()
                    return

                self.state = self.State.SETUP_SERVICE
            elif value == _MODEL_LOADED_MESSAGE:
                if self.state != self.State.SETUP_SERVICE:
                    _LOG.error(
                        "_read_predict_loop_startup_message: %s message: "
                        "not in expected state: %s",
                        value,
                        self.state,
                    )
                    _remove_reader()
                    return

                self.state = self.State.RUNNING
                self._predict_loop_watchdog_task = asyncio.create_task(
                    self._predict_loop_watchdog(  # noqa
                        _PREDICT_LOOP_WATCHDOG_SECONDS, self._check_predict_loop  # noqa
                    )
                )
                _remove_reader()

        asyncio.get_event_loop().add_reader(read_conn.fileno(), read_startup_message)
        self._request_queue.put(write_conn)

    def _stop_predict_loop(self):
        if not self._request_queue:
            return

        if self._predict_loop_watchdog_task is not None:
            self._predict_loop_watchdog_task.cancel()
        self._request_queue.put(_TERMINATE_PREDICT_LOOP_REQUEST)
        self._predict_process.join()
        self._request_queue = None

    async def _predict_loop_watchdog(self, interval, periodic_function):
        while True:
            await asyncio.gather(
                asyncio.sleep(interval),
                periodic_function(),
            )

    def get_api_schema(self) -> Dict[str, Any]:
        """Return the Open API schema for the underlying service."""
        return self.app.openapi()

    def get_usage_examples(
        self, format: documenter.DocumentationFormat
    ) -> MustacheGenerationResult:
        """Return the Mustache generation result for the underlying service."""
        return documenter.generate_usage_examples(self.service, format)

    def store_assets(self) -> None:
        """Run service store assets."""
        self.service.store_assets()

    def run(self, port: int, timeout_keep_alive: int):
        """Run the server exposing the underlying service."""
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=port,
            timeout_keep_alive=timeout_keep_alive,
            lifespan="on",
        )


def load_service(module_name: str, class_name: Optional[str] = None) -> Service:
    """Load a class from service implementation."""
    try:
        module = importlib.import_module(module_name)

        if class_name is not None:
            # if service class is provided, instantiate it
            class_ = getattr(module, class_name)
        else:
            # if service class not provided, look for it
            class_ = None
            for name, class_obj in inspect.getmembers(module, inspect.isclass):
                for class_base in class_obj.__bases__:
                    if (
                        class_base.__module__ == _OCTOAI_SERVICE_MODULE
                        and class_base.__name__ == _OCTOAI_BASE_SERVICE_CLASS
                    ):
                        class_ = class_obj
                        break

            if class_ is None:
                raise ValueError(
                    f"Module '{module_name}' contains no classes extending "
                    f"base '{_OCTOAI_SERVICE_MODULE}.{_OCTOAI_BASE_SERVICE_CLASS}'"
                )

        _LOG.info(f"Using service in {module_name}.{class_.__name__}.")

        return class_()
    except ModuleNotFoundError:
        error_msg = f"Module '{module_name}' not found. "
        if module_name == "service":
            error_msg += "Ensure your service is defined in service.py."
        raise ValueError(error_msg)


def _load_server(ctx, async_enable: bool = True) -> Server:
    service = load_service(
        ctx.parent.params["service_module"],
        class_name=ctx.parent.params["service_class"],
    )
    return Server(service, async_enable=async_enable)


@click.group(name="server")
@click.option(
    "--log-level",
    type=click.Choice(["ERROR", "WARNING", "INFO", "DEBUG"], case_sensitive=False),
    default="INFO",
    envvar="OCTOAI_LOG_LEVEL",
)
@click.option("--service-module", default="service")
@click.option("--service-class", default=None)
@click.pass_context
def server(ctx, log_level, service_module, service_class):
    """CLI for OctoAI server."""
    logging.basicConfig(
        level=log_level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
    )
    click.echo("octoai server")
    ctx.ensure_object(dict)


@server.command()
@click.option("--output-file", default=None)
@click.pass_context
def api_schema(ctx, output_file):
    """Generate OpenAPI schema for the given service."""
    _LOG.info("api-schema")
    server: Server = _load_server(ctx)
    schema = server.get_api_schema()

    if output_file:
        with open(output_file, "w") as f:
            json.dump(schema, f, indent=2)
    else:
        click.echo(json.dumps(schema, indent=2))


@server.command(hidden=True)
@click.option("--no-render", is_flag=True, default=False)
@click.option(
    "--format",
    required=True,
    type=click.Choice(["curl", "python"], case_sensitive=False),
)
@click.option("--output-file", default=None)
@click.pass_context
def generate_usage_examples(ctx, no_render, format, output_file):
    """Generate client usage examples for the given service."""
    _LOG.info("generate-usage-examples")

    def convert_format(f: str) -> documenter.DocumentationFormat:
        if f.lower() == "curl":
            return documenter.DocumentationFormat.CURL
        elif f.lower() == "python":
            return documenter.DocumentationFormat.PYTHON
        else:
            raise ValueError(f"format '{f}' is not supported")

    server: Server = _load_server(ctx)
    generation_result = server.get_usage_examples(convert_format(format))

    if not no_render:
        rendered_result = chevron.render(
            generation_result.generated_template, generation_result.template_hash
        )

        if output_file:
            with open(output_file, "w") as f:
                f.write(rendered_result)
        else:
            click.echo(rendered_result)
    else:
        if output_file:
            with open(output_file, "w") as f:
                json.dump(dataclasses.asdict(generation_result), f, indent=2)
        else:
            click.echo(json.dumps(dataclasses.asdict(generation_result), indent=2))


@server.command()
@click.pass_context
def setup_service(ctx):
    """Run the setup code for the given service."""
    _LOG.info("setup-service")
    server: Server = _load_server(ctx, async_enable=False)
    server.service.setup()


@server.command()
@click.pass_context
@click.option("--port", type=int, default=8080)
@click.option("--async-enable", default=True)
@click.option(
    "--timeout-keep-alive",
    type=int,
    default=900,
    help="Connection keep alive timeout in seconds",
)
def run(ctx, port, async_enable, timeout_keep_alive):
    """Run the server for the given service."""
    _LOG.info("run")
    maybe_set_volume_environment_variables(ctx)
    server: Server = _load_server(ctx, async_enable=async_enable)
    server.run(port, timeout_keep_alive)


@server.command()
@click.pass_context
@click.option("--check-is-needed", is_flag=True)
def store_assets(ctx, check_is_needed):
    """Run the store_assets code for the given model."""
    _LOG.info("store_assets")
    if check_is_needed:
        sys.exit(
            _STORE_ASSET_NEEDED_CODE
            if is_store_assets_needed(ctx)
            else _STORE_ASSET_NOT_NEEDED_CODE
        )
    maybe_set_volume_environment_variables(ctx)
    server: Server = _load_server(ctx, async_enable=False)
    server.store_assets()


if __name__ == "__main__":
    server(obj={})
