// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "doctags_parser.hpp"
#include "markdown_parser.hpp"
#include "plain_text_parser.hpp"

namespace cv {

namespace {

String modelName(vlm::VLMModelType model_type, const String& model_dir)
{
    switch (model_type)
    {
        case vlm::VLM_MODEL_PADDLEOCR_VL:    return "paddleocr-vl-1.5";
        case vlm::VLM_MODEL_GRANITE_DOCLING: return "granite-docling-258m";
        default:                             return model_dir;
    }
}

String engineName(const String& engine)
{
    // vlm::create() only accepts "opencv" today; ENGINE_ORT cannot carry the KV cache
    // these decoders need. Kept as a lookup so adding engines here stays a one-liner.
    return engine == "opencv" ? "ENGINE_OPENCV" : engine;
}

// Each vlm engine has its own internal default prompt (for callers using the vlm module
// directly), but docRead() must not rely on falling through to it -- prompt tuning for
// docRead() lives here, in docProc, not in the vlm module.
String defaultPromptFor(vlm::VLMModelType model_type)
{
    switch (model_type)
    {
    case vlm::VLM_MODEL_GRANITE_DOCLING:
        return "Convert this page to docling. Preserve OCR text, table structure, "
               "form fields, and layout/section structure.";
    case vlm::VLM_MODEL_PADDLEOCR_VL:
        // PaddleOCR-VL only supports four fixed task prompts and needs pre-cropped,
        // single-purpose regions for genuine structure (see official docs); "OCR:" is
        // its documented whole-page prompt, left to parsePlainTextPage's heuristics.
        return "OCR:";
    case vlm::VLM_MODEL_OPENAI:
    case vlm::VLM_MODEL_ANTHROPIC:
    case vlm::VLM_MODEL_GEMINI:
    case vlm::VLM_MODEL_GROK:
        // General-purpose instruction-following models: unlike PaddleOCR-VL, asking for
        // real Markdown structure on a whole page is well within their normal capability.
        return "Convert this page to Markdown. Use # headers for titles/sections, and "
               "GitHub-flavored Markdown tables (with | and a --- separator row) for any "
               "tabular data. Preserve all text content and layout structure.";
    default:
        return String();
    }
}

} // namespace

docproc::DocumentResult docRead(const String& input_path, vlm::VLMModelType model_type,
                                 const String& model_dir, const String& engine,
                                 const String& device, const String& api_key, const String& prompt,
                                 int max_new_tokens, bool raw)
{
    Mat image = imread(input_path, IMREAD_COLOR);
    if (image.empty())
        CV_Error(Error::StsError, "docRead: failed to read input image: " + input_path);

    Ptr<vlm::VLMModel> model = vlm::create(model_type, model_dir, engine, device, api_key);

    String actualPrompt = prompt.empty() ? defaultPromptFor(model_type) : prompt;

    TickMeter timer;
    timer.start();
    // Bypasses inferDocument() to reuse the Mat already decoded above; only valid while
    // inferDocument() treats one image path as exactly one page (see vlm.hpp) -- revisit
    // if it grows multi-page input support.
    model->reset();
    std::vector<String> rawPages = { model->infer(image, actualPrompt, max_new_tokens) };
    timer.stop();

    docproc::DocumentResult result;
    result.metadata.model = modelName(model_type, model_dir);
    result.metadata.engine = engineName(engine);
    result.metadata.pages = (int)rawPages.size();
    result.metadata.tokens_used = model->lastTokensUsed();
    result.metadata.inference_time_ms = (int64)timer.getTimeMilli();

    if (raw)
    {
        result.raw_pages = rawPages;
        return result;
    }

    result.pages.resize(rawPages.size());
    for (size_t i = 0; i < rawPages.size(); ++i)
    {
        docproc::Page& page = result.pages[i];
        page.page_number = (int)i + 1;
        page.width = image.cols;
        page.height = image.rows;

        if (docproc::isDocTags(rawPages[i]))
        {
            docproc::parseDocTagsPage(rawPages[i], page);
        }
        else if (docproc::isOtslCellStream(rawPages[i]))
        {
            docproc::parseOtslCellStream(rawPages[i], page);
        }
        else if (docproc::isMarkdown(rawPages[i]))
        {
            docproc::parseMarkdownPage(rawPages[i], page);
        }
        else
        {
            docproc::parsePlainTextPage(rawPages[i], page);
        }
    }

    return result;
}

} // namespace cv
