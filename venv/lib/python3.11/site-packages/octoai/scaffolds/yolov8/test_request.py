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
import json

from octoai.client import Client
from octoai.types import Image

image = Image.from_url(
    url="http://ultralytics.com/images/bus.jpg",
    b64=True,
    follow_redirects=True,
)
inputs = {"image": image.model_dump()}


def main(endpoint):
    """Run inference against the endpoint."""
    # create an OctoAI client
    client = Client()

    # perform inference
    response = client.infer(endpoint_url=f"{endpoint}/infer", inputs=inputs)

    # show coordinates for detections
    output = response["output"]
    print("response.output.detections:")
    print(json.dumps(output["detections"], indent=2))

    # save image with detections rendered on top of input image
    output_file = "detections.jpg"
    print("response.output.image_with_boxes:")
    detections = Image.from_endpoint_response(output, key="image_with_boxes")
    print(f"Saving as {output_file}")
    detections.to_file(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, default="http://localhost:8080")

    args = parser.parse_args()
    main(args.endpoint)
