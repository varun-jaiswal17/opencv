// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DOCPROC_MARKDOWN_PARSER_HPP
#define OPENCV_DOCPROC_MARKDOWN_PARSER_HPP

#include "opencv2/docproc.hpp"

namespace cv { namespace docproc {

/** @brief True if raw contains a Markdown `#` header line or a `|`-pipe table (header row
followed by a `---` separator row), as commercial VLM APIs (OpenAI/Anthropic/Gemini/Grok)
tend to produce when asked for structured output. */
bool isMarkdown(const String& raw);

/** @brief Parses Markdown into page.blocks: `#`..`######` header lines become BLOCK_TITLE,
`|`-pipe tables (header + `---` separator + data rows) become BLOCK_TABLE, blank-line
separated runs of anything else become BLOCK_TEXT. Inline emphasis/list/link markup is left
as-is in the text -- only headers and tables are structurally recognized.
*/
void parseMarkdownPage(const String& raw, Page& page);

}} // namespace cv::docproc

#endif // OPENCV_DOCPROC_MARKDOWN_PARSER_HPP
