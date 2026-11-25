"""OpenCV MCP server: docRead, detect and segment over OpenCV DNN models, for any MCP client.

    python server.py                       HTTP on http://127.0.0.1:8000/mcp, one shared server
    python server.py --stdio               for an app that starts the server itself
    python server.py --allow ~/Pictures    let the tools read images from that folder too

Images are passed by path, and the tools read only files inside the allowed folders: the
uploads folder next to this file, and every folder given with --allow. Each tool checks its
inputs and passes the work to runner.run_model(). Which engine and target a model runs on
comes from models.yml, never from a tool call.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2 as cv
from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image
from mcp.types import TextContent

import registry
import results
import runner

HERE = Path(__file__).resolve().parent
ALLOWED = [HERE / "uploads"]   # the folders the tools may read images from
OUTPUTS = HERE / "outputs"     # where segment saves its masks
PICTURE_SIZE = 1024            # longest side of the picture that annotate=true returns

mcp = FastMCP("opencv", instructions="OpenCV computer vision tools. Pass images as file paths inside the "
                                     "server's allowed folders; call list_models to see which models each tool can use.")
registry.load_registry()


def safe_path(path):
    """The full path of an image, refusing anything outside the allowed folders. A relative path
    may start inside an allowed folder ("street.jpg") or name it ("uploads/street.jpg")."""
    path = Path(path).expanduser()
    for folder in (f.resolve() for f in ALLOWED):
        for base in (folder, folder.parent):
            full = (base / path).resolve()
            if full.is_relative_to(folder) and full.is_file():
                return str(full)
    allowed = ", ".join(str(f.resolve()) for f in ALLOWED)
    raise ValueError(f"{path} is not a file in the allowed folders ({allowed}); "
                     "copy the image there, or start the server with --allow <folder>")


def reply(task, found, image_path, model, annotate):
    """The JSON of a result and, with annotate, a picture of it drawn on the image."""
    out = results.to_json(task, found, image_path, model, OUTPUTS)
    if not annotate:
        return out
    picture = results.draw(task, found, cv.imread(image_path))
    scale = PICTURE_SIZE / max(picture.shape[:2])
    if scale < 1:
        picture = cv.resize(picture, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    png = cv.imencode(".png", picture)[1].tobytes()
    return ToolResult(content=[TextContent(type="text", text=json.dumps(out)),
                               Image(data=png, format="png").to_image_content()],
                      structured_content=out)


@mcp.tool()
def docRead(image_path: str, model: str = "granite", prompt: str = "") -> dict:
    """Read a document page (a scan, photo or screenshot) and return its text as Markdown.
    image_path: the path of an image file in one of the server's allowed folders.
    model: "granite" (whole pages, tables, lists) or "paddleocr_vl". Call list_models for all options.
    prompt: leave empty to read the whole page. paddleocr_vl also accepts
    "Table Recognition:", "Formula Recognition:" and "Chart Recognition:"."""
    image = safe_path(image_path)
    return results.to_json("docRead", runner.run_model(model, "docRead", image, prompt=prompt), image, model, OUTPUTS)


@mcp.tool()
def detect(image_path: str, model: str = "yoloe26", classes: list[str] | None = None,
           score_threshold: float | None = None, annotate: bool = False) -> dict:
    """Find objects in an image. Returns each object's box [x, y, width, height] in pixels,
    its class name and a score from 0 to 1.
    image_path: the path of an image file in one of the server's allowed folders.
    model: "yoloe26" (fast, the 80 COCO classes such as person, car, dog) or "owlv2"
    (any objects you name, such as "red umbrella", but slower).
    classes: the names to look for, such as ["car", "person"]. owlv2 needs them; for yoloe26,
    leave empty to get all classes.
    score_threshold: hide objects with a lower score; leave empty for the model's default.
    annotate: also return a picture of the image with the boxes drawn on it."""
    image = safe_path(image_path)
    found = runner.run_model(model, "detect", image, classes=classes, score_threshold=score_threshold)
    return reply("detect", found, image, model, annotate)


@mcp.tool()
def segment(image_path: str, model: str = "sam3", prompt: str | None = None,
            reference_image: str | None = None, reference_mask: str | None = None,
            annotate: bool = False) -> dict:
    """Cut objects out of an image. Saves each mask as a PNG file (white = object) and returns
    its path, its box [x, y, width, height] and its area in pixels.
    image_path: the path of an image file in one of the server's allowed folders.
    model "sam3": give a prompt describing what to cut out, such as "dog".
    model "insid3": give reference_image and reference_mask instead: another image with the
    same kind of object marked in white on black, also in an allowed folder.
    annotate: also return a picture of the image with the masks drawn on it."""
    image = safe_path(image_path)
    if reference_image:
        reference_image = safe_path(reference_image)
    if reference_mask:
        reference_mask = safe_path(reference_mask)
    masks = runner.run_model(model, "segment", image, prompt=prompt,
                             reference_image=reference_image, reference_mask=reference_mask)
    return reply("segment", masks, image, model, annotate)


@mcp.tool()
def list_models() -> dict:
    """List the models of each tool, what each is good at, and whether it can run on this server."""
    return runner.list_models()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stdio", action="store_true", help="talk over stdin/stdout instead of HTTP")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--allow", type=Path, action="append", default=[], metavar="FOLDER",
                        help="also let the tools read images from this folder; can be given more than once")
    parser.add_argument("--outputs", type=Path, default=OUTPUTS, help="the folder segment saves masks to")
    args = parser.parse_args()
    reason = registry.forced_engine()
    if reason:
        sys.exit(reason)
    ALLOWED += [f.expanduser() for f in args.allow]
    OUTPUTS = args.outputs
    ALLOWED[0].mkdir(parents=True, exist_ok=True)
    if args.stdio:
        mcp.run()
    else:
        # 127.0.0.1 only: anyone who can reach the port can run the tools
        mcp.run(transport="http", host="127.0.0.1", port=args.port)
