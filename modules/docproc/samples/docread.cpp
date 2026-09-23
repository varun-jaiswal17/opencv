// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
 * Sample program for cv::docRead(): runs a VLM OCR / document-understanding engine over a
 * page image and returns a unified, engine-agnostic result -- pages made of typed blocks
 * (title/text/table/form/signature), each carrying lines, table cells or form fields.
 *
 * Local model types (paddleocr_vl, granite_docling) load ONNX weights from --model_dir,
 * following the upstream layout of the model's ONNX export:
 *     https://huggingface.co/PaddlePaddle/PaddleOCR-VL
 *     https://huggingface.co/onnx-community/granite-docling-258M-ONNX
 * Cloud model types (openai, anthropic, gemini, grok) call the provider's hosted vision API
 * instead -- pass the provider's model name as --model_dir and an API key via --api_key, or
 * via the provider's usual environment variable (OPENAI_API_KEY, ANTHROPIC_API_KEY,
 * GEMINI_API_KEY, GROK_API_KEY).
 *
 * Run the sample:
 *
 *      ./docread --model_type=granite_docling \
 *                --model_dir=<path-to-granite-docling-258M-ONNX-export> \
 *                --input=<path-to-page-image>
 *
 *      ./docread --model_type=openai --model_dir=gpt-4o --input=<path-to-page-image>
 */

#include <cstdlib>
#include <iostream>
#include <map>

#include <opencv2/docproc.hpp>
#include <opencv2/vlm.hpp>

using namespace cv;
using namespace std;

static const map<String, vlm::VLMModelType> MODEL_TYPES = {
    { "paddleocr_vl",    vlm::VLM_MODEL_PADDLEOCR_VL },
    { "granite_docling", vlm::VLM_MODEL_GRANITE_DOCLING },
    { "openai",          vlm::VLM_MODEL_OPENAI },
    { "anthropic",       vlm::VLM_MODEL_ANTHROPIC },
    { "gemini",          vlm::VLM_MODEL_GEMINI },
    { "grok",            vlm::VLM_MODEL_GROK },
};

static const map<String, String> API_KEY_ENV_VARS = {
    { "openai",    "OPENAI_API_KEY" },
    { "anthropic", "ANTHROPIC_API_KEY" },
    { "gemini",    "GEMINI_API_KEY" },
    { "grok",      "GROK_API_KEY" },
};

static const char* BLOCK_TYPE_NAMES[] = { "title", "text", "table", "form", "signature" };

static void printBlock(const docproc::Block& block, int indent)
{
    string prefix(indent, ' ');
    cout << prefix << "[" << BLOCK_TYPE_NAMES[block.type] << "]" << endl;
    for (const docproc::Line& line : block.lines)
        cout << prefix << "  " << line.text << endl;
    for (const docproc::Cell& cell : block.cells)
        cout << prefix << "  [" << cell.row << "," << cell.col << "]"
             << (cell.is_header ? " (header) " : " ") << cell.text << endl;
    for (const docproc::Field& field : block.fields)
        cout << prefix << "  " << field.key << ": " << field.value << endl;
}

static void printResult(const docproc::DocumentResult& result)
{
    const docproc::DocumentMetadata& m = result.metadata;
    cout << "model=" << m.model << " engine=" << m.engine << " pages=" << m.pages
         << " tokens_used=" << m.tokens_used
         << " inference_time_ms=" << m.inference_time_ms << endl;
    for (const docproc::Page& page : result.pages)
    {
        cout << "--- page " << page.page_number << " (" << page.width << "x" << page.height
             << ") ---" << endl;
        for (const docproc::Block& block : page.blocks)
            printBlock(block, 2);
    }
}

int main(int argc, char** argv)
{
    const string keys =
        "{ help h         |        | Print help message }"
        "{ model_type     |        | Which VLM engine to use: paddleocr_vl, granite_docling, "
                                     "openai, anthropic, gemini or grok }"
        "{ model_dir      |        | Local model types: path to the local ONNX export directory. "
                                     "Cloud model types: the provider model name, e.g. gpt-4o }"
        "{ input i        |        | Path to the input page image }"
        "{ engine         | opencv | cv::dnn engine for local model types; only \"opencv\" is "
                                     "supported; ignored by cloud model types }"
        "{ device         | cpu    | Compute device for local model types (cpu or cuda); ignored "
                                     "by cloud model types }"
        "{ api_key        |        | API key for cloud model types; falls back to the provider's "
                                     "usual environment variable when omitted }"
        "{ prompt         |        | Task prompt; empty uses the engine's default prompt }"
        "{ max_new_tokens | 512    | Maximum number of tokens the engine may generate }"
        "{ raw            |        | Skip structured parsing and print each page's raw engine "
                                     "output }";

    CommandLineParser parser(argc, argv, keys);
    parser.about("Run cv::docRead() on a page image.");
    if (parser.has("help") || !parser.has("model_type") || !parser.has("model_dir") ||
        !parser.has("input"))
    {
        parser.printMessage();
        return 0;
    }

    string modelTypeName = parser.get<String>("model_type");
    map<String, vlm::VLMModelType>::const_iterator modelTypeIt = MODEL_TYPES.find(modelTypeName);
    if (modelTypeIt == MODEL_TYPES.end())
    {
        cerr << "Unknown --model_type: " << modelTypeName << endl;
        return 1;
    }

    string apiKey = parser.get<String>("api_key");
    if (apiKey.empty())
    {
        map<String, String>::const_iterator envIt = API_KEY_ENV_VARS.find(modelTypeName);
        if (envIt != API_KEY_ENV_VARS.end())
        {
            const char* fromEnv = getenv(envIt->second.c_str());
            if (fromEnv)
                apiKey = fromEnv;
        }
    }

    bool raw = parser.has("raw");
    docproc::DocumentResult result = docRead(parser.get<String>("input"), modelTypeIt->second,
                                              parser.get<String>("model_dir"),
                                              parser.get<String>("engine"),
                                              parser.get<String>("device"), apiKey,
                                              parser.get<String>("prompt"),
                                              parser.get<int>("max_new_tokens"), raw);

    if (raw)
    {
        for (size_t i = 0; i < result.raw_pages.size(); i++)
        {
            cout << "--- page " << (i + 1) << " (raw) ---" << endl;
            cout << result.raw_pages[i] << endl;
        }
    }
    else
    {
        printResult(result);
    }

    return 0;
}
