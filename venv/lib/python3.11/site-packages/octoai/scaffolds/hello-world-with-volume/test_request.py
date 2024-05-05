"""
test_request.py.

This program makes an example request to your endpoint.
You typically use this program to verify that your service
works correctly:

- After `octoai run` (local development).
  No extra arguments are needed in this case.

  Example:
  `octoai run --command "python test_request.py"`

- After `octoai deploy` (remote deployment).
  Provide your endpoint's host name and port as arguments.

  Example:
  `python test_request.py --endpoint https://myapp123.octoai.cloud`

This program also serves as basic example on how to use the
Python client of the OctoAI SDK. As you develop your service,
you may have to change the inputs in this program to match
what your service expects.
"""
import argparse

from octoai.client import Client

inputs = {"prompt": "Hello world!"}


def main(endpoint):
    """Run inference against the endpoint."""
    # create an OctoAI client
    client = Client()

    # perform inference
    response = client.infer(endpoint_url=f"{endpoint}/infer", inputs=inputs)

    # show the response
    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, default="http://localhost:8080")

    args = parser.parse_args()
    main(args.endpoint)
