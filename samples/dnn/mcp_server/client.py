"""Call the OpenCV MCP server from a terminal, without an AI app.

    python client.py list_models
    python client.py detect image_path=street.jpg classes='["car", "person"]'
    python client.py segment image_path=messi5.jpg model=sam3 prompt=ball

Each argument is name=value. A value that reads as JSON (a number, a list, true/false) is
passed as that, anything else as text. The server must be running: python server.py
"""

import argparse
import asyncio
import json

from fastmcp import Client


def value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


async def call(url, tool, args):
    async with Client(url) as client:
        if tool == "tools":
            for t in await client.list_tools():
                print(f"{t.name}: {t.description}\n")
            return
        result = await client.call_tool(tool, args, raise_on_error=False)
        if result.is_error:
            raise SystemExit("error: " + result.content[0].text)
        print(json.dumps(result.structured_content, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tool", help="docRead, detect, segment, list_models, or 'tools' to describe them")
    parser.add_argument("args", nargs="*", metavar="name=value")
    parser.add_argument("--url", default="http://127.0.0.1:8000/mcp")
    args = parser.parse_args()
    pairs = {}
    for a in args.args:
        name, sep, text = a.partition("=")
        if not sep:
            parser.error(f"'{a}' is not name=value")
        pairs[name] = value(text)
    asyncio.run(call(args.url, args.tool, pairs))
