# OpenCV MCP server

A local server that offers OpenCV DNN models as tools over MCP (Model Context Protocol), the
protocol AI apps use to call tools. Any MCP client can use it: an AI app such as Claude, VS Code,
Cursor, Gemini CLI or LM Studio, the small `client.py` in this folder, or your own program.

```
 AI app or client.py ──MCP──► server.py ──► runner.py ──► adapters/<model>.py ──► cv2.dnn / cv2.vlm
                               4 tools       loads each      prepares the input,
                                             model once      runs it, reads the output
```

| Tool | Does | Models |
|---|---|---|
| `docRead` | page image → Markdown | `granite`, `paddleocr_vl` |
| `detect` | image → boxes, class names, scores | `yoloe26`, `owlv2` |
| `segment` | image → masks saved as PNG files | `sam3`, `insid3` |
| `list_models` | what each tool can use, and whether it can run here | |

The engine and target of each model are set in `models.yml`, never by a tool call.
`validate.py` runs every model on `ENGINE_OPENCV` and `ENGINE_OPENVINO` and compares the results.

## Setup

1. Build OpenCV 5.x with OpenVINO and the Python bindings (add `vlm` for `docRead`, from opencv/opencv#30146):

   ```bash
   cmake -S opencv -B build -DWITH_OPENVINO=ON -DOpenVINO_DIR=<openvino>/runtime/cmake \
         -DBUILD_LIST=core,imgproc,imgcodecs,dnn,python3
   cmake --build build -j
   export PYTHONPATH=$PWD/build/python_loader
   ```

2. Install the Python packages: `pip install -r requirements.txt`

3. Put the models in one folder and point `OPENCV_DOWNLOAD_CACHE_DIR` at it, as for the other DNN
   samples. `models.yml` names each file relative to that folder:

   ```
   $OPENCV_DOWNLOAD_CACHE_DIR/
     granite-docling-258M/     the export folder described in cv::vlm::create()
     paddleocr-vl/             the same, for PaddleOCR-VL
     yoloe26/yoloe-26s-seg-nms.onnx
     owlv2/owlv2-base-patch16.onnx, owlv2/tokenizer.json
     sam3/sam3_image_encoder.onnx, sam3_language_encoder.onnx, sam3_decoder.onnx, tokenizer.json
     insid3/dinov3_vits16.onnx, insid3/model.onnx_data
   ```

| Model | Where to get it |
|---|---|
| `granite` | [onnx-community/granite-docling-258M-ONNX](https://huggingface.co/onnx-community/granite-docling-258M-ONNX), plus the `config.json` that `cv::vlm::create()` describes |
| `paddleocr_vl` | as described in `cv::vlm::create()` |
| `yoloe26` | export it with Ultralytics, with the classes of `labels/yoloe26.txt` in that order (below) |
| `owlv2` | `onnx/model.onnx` and `tokenizer.json` of [onnx-community/owlv2-base-patch16-finetuned-ONNX](https://huggingface.co/onnx-community/owlv2-base-patch16-finetuned-ONNX) |
| `sam3` | the three FP32 files of [wkentaro/sam3-onnx-models-v0.7.0](https://huggingface.co/wkentaro/sam3-onnx-models-v0.7.0) (SAM License), and `tokenizer.json` of [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) |
| `insid3` | `onnx/model.onnx` and `onnx/model.onnx_data` of [onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX](https://huggingface.co/onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX), renamed to `dinov3_vits16.onnx`; keep the two files in one real folder, because OpenVINO refuses external data reached through a symbolic link |

YOLOE-26 export:

```python
from ultralytics import YOLOE
model = YOLOE("yoloe-26s-seg.pt")
model.set_classes(open("labels/yoloe26.txt").read().splitlines())
model.export(format="onnx", end2end=False)    # then rename to yoloe-26s-seg-nms.onnx
```

`end2end=False` keeps the output that needs NMS. OpenVINO 2026.4 can't compile the default
end-to-end head: its `GatherElements` shape check rejects the shapes that ONNX allows.

## Run

```bash
python server.py                       # http://127.0.0.1:8000/mcp, one shared server; models stay loaded
python server.py --allow ~/Pictures    # let the tools read images from that folder too
python server.py --stdio               # for an app that starts the server itself
```

Images are passed by file path, so the client must run on the same machine. The tools read only
files inside the allowed folders: the `uploads` folder next to `server.py`, and each folder given
with `--allow` (repeat it for more folders). A path may be absolute, or relative to an allowed
folder (`street.png`, `uploads/street.png`). An image pasted into a chat has no path, so save it
to a file first. `segment` writes its masks to `outputs`.

`detect` and `segment` take `annotate=true` to also return a picture with the boxes or masks
drawn on it, as MCP image content, so that an AI app can look at the result.

The server refuses to start while `OPENCV_FORCE_DNN_ENGINE` is set, because that variable
replaces the engine that `models.yml` gives each model.

## Connect a client

Every app needs a name and the URL. Only the file differs:

| App | File | Entry |
|---|---|---|
| Claude Code | command | `claude mcp add --transport http opencv http://127.0.0.1:8000/mcp` |
| VS Code | `.vscode/mcp.json` | `{"servers": {"opencv": {"type": "http", "url": "http://127.0.0.1:8000/mcp"}}}` |
| Cursor | `~/.cursor/mcp.json` | `{"mcpServers": {"opencv": {"url": "http://127.0.0.1:8000/mcp"}}}` |
| Gemini CLI | `settings.json` | `{"mcpServers": {"opencv": {"httpUrl": "http://127.0.0.1:8000/mcp"}}}` |
| LM Studio | `mcp.json` | as for Cursor; with a local model, nothing leaves the machine |

An app that starts the server itself takes the `--stdio` form instead. Such apps start it with
few environment variables, so pass the two it needs:

```json
{"mcpServers": {"opencv": {
    "command": "python",
    "args": ["/path/to/mcp_server/server.py", "--stdio"],
    "env": {"PYTHONPATH": "/path/to/build/python_loader", "OPENCV_DOWNLOAD_CACHE_DIR": "/path/to/models"}}}}
```

Without an AI app, call a tool from a terminal:

```bash
python client.py list_models
python client.py detect image_path=street.png classes='["car", "person"]'
python client.py segment image_path=messi5.jpg prompt=ball
python client.py segment image_path=~/Pictures/dog.jpg prompt=dog    # with --allow ~/Pictures
python client.py tools          # the tool descriptions an AI app sees
```

`granite` takes a few minutes for a dense page on `ENGINE_OPENCV`; raise the tool timeout of
your client if it gives up earlier.

## Validate the engines

```bash
python validate.py --testset /path/to/testset            # every model
python validate.py --testset /path/to/testset --models yoloe26,owlv2
```

The test set has a folder per tool (`docRead/`, `detect/`, `segment/`) with that tool's images.
`test_inputs` in `models.yml` gives what each model gets besides the image, e.g. the classes for
`owlv2`; a file named there, such as the reference image of `insid3`, is relative to the test set.
The default test set is `$OPENCV_TEST_DATA_PATH/dnn/mcp`.

The engines pass when text differs by a character error rate of at most 0.05, boxes match one
to one with IoU ≥ 0.9 and scores within 0.02, and masks match with IoU ≥ 0.95. A model that one
engine can't run is reported as SKIP, with the reason. `validate.py` refuses to start while
`OPENCV_FORCE_DNN_ENGINE` is set: both runs would come from one engine.

## Tests

```bash
python -m pytest tests
```

The tests use a fake adapter and need no models; the tokenizer test runs only when a CLIP
`tokenizer.json` is in `OPENCV_DOWNLOAD_CACHE_DIR`.

## Add a model

1. Add an entry to `models.yml` (and a labels file if its classes are fixed).
2. Write `adapters/<name>.py`, a subclass of `Adapter` with `run()`, or reuse an adapter for
   another size of the same model.
3. Add it to `ADAPTERS` in `adapters/__init__.py`.
4. Add test images, and run `validate.py --models <name>`.

`server.py`, `runner.py` and `registry.py` don't change.

## Known limits

- `docRead` runs on `ENGINE_OPENCV` only: `cv2.vlm` needs a KV cache, which `ENGINE_OPENVINO` doesn't have yet.
- `adapters/clip_tokenizer.py` encodes the prompts of `owlv2` and `sam3`, because `cv.dnn.Tokenizer`
  does not handle CLIP's `</w>` end-of-word marker yet.
- `ENGINE_OPENCV` can't load the DINOv3 export of `insid3` (no `SplitToSequence`).
- OmDet-Turbo is not here yet: it needs an ONNX export first.
- Only local clients: an app that reaches MCP servers over the internet only, such as ChatGPT,
  needs the server behind HTTPS with a login and a way to upload images.
