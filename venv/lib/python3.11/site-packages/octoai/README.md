# octoai

This folder contains the implementation of the Python base **server** that runs on each container. The server provides all the plumbing needed to provide a REST API to a client that wants to connect to the container. The server supports (i.e. runs) a model-specific prediction **service** that contains the user's actual model and logic.

## Installation
You can run `poetry install`, which will use the dependencies described in `pyproject.toml`. However, this means that all the dependencies are associated with `poetry` rather than `python`. The implications of this are described in the next section.

## Running the server
Whenever you make a change to the server, it's a good idea to try running the server locally to see if your changes broke anything.

### For scaffold dev iteration
Run each of the following from `cli-go/cmd/server`.

Note that if you installed dependencies using `poetry install`, then you'll likely get several missing dependency complaints. You'll need to add `poetry run` in front of each command so that it uses the poetry dependencies instead.

#### To generate API schema
This will provide a json representation of the available endpoints for this service. It is the same as what is provided when you visit `localhost:<port>/openapi.json` in the browser.
```shell
python -m octoai.server \
  --service-module scaffolds.flan_t5.service \
  --service-class T5Service \
  api-schema
```

#### Initialize service only
This will run the `setup` method for the service only, however it is defined.
```shell
python -m octoai.server \
  --service-module scaffolds.flan_t5.service \
  --service-class T5Service \
  setup-service
```

#### Run the server
This will allow you to actually make inferencecs against the service.
```shell
python -m octoai.server \
  --service-module scaffolds.flan_t5.service \
  --service-class T5Service \
  run \
  --port 8000
```

### From a scaffold directory
If you're running somewhere where `service.py` is in the current dir, then you won't need any extra parameters. Instead each command will look like
```shell
python -m octoai.server api-schema

python -m octoai.server setup-service

python -m octoai.server run --port 8000
```
The same caveat stated above about dependencies still holds in this case.

## Testing
Run `make test` from the `cli/cmd/server` folder to run all automated unit and integration tests for the server.

Run `make build` from the `cli/` folder to build an executable, which will show up in the `dist/` folder. You can then see how the CLI behaves in its current iteration by running `./octoai <cmd>` (or just `octoai`, if you added the executable to your `PATH`).

Note that there are two different Makefiles with different purposes. TODO: We plan to merge these Makefiles, but at the moment we'll continue to have one for the CLI and one for Python.

# scaffolds
This directory contains several starter templates for services off of which users can base their own service implementations. Each scaffold requires a `service.py` that defines the inference schema that overrides the base service provided by the server.
