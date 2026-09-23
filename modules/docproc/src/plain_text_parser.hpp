// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DOCPROC_PLAIN_TEXT_PARSER_HPP
#define OPENCV_DOCPROC_PLAIN_TEXT_PARSER_HPP

#include "opencv2/docproc.hpp"

namespace cv { namespace docproc {

/** @brief Groups plain OCR text (no layout markup, e.g. PaddleOCR-VL's default output) into
Page.blocks: runs of >=2 consecutive lines that all split into the same number of
whitespace-aligned columns become a BLOCK_TABLE, blank-line-separated runs of any other
lines become a BLOCK_TEXT. There is no positional signal in plain text, so all bbox
fields are left at their default (empty) value.
*/
void parsePlainTextPage(const String& raw, Page& page);

}} // namespace cv::docproc

#endif // OPENCV_DOCPROC_PLAIN_TEXT_PARSER_HPP
