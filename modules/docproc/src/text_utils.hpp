// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DOCPROC_TEXT_UTILS_HPP
#define OPENCV_DOCPROC_TEXT_UTILS_HPP

#include "opencv2/docproc.hpp"

#include <functional>
#include <vector>

namespace cv { namespace docproc {

String trimmed(const String& s);

std::vector<String> splitLines(const String& raw);

//! Builds a BLOCK_TEXT block from accumulated lines; clears lines on return.
Block makeTextBlock(std::vector<Line>& lines);

//! Fills block.cells from a rectangular grid of cell text, row-major; isHeaderRow(r)
//! marks header cells for row r. Caller is responsible for setting block.rows/cols.
void fillTableCells(Block& block, const std::vector<std::vector<String>>& rows,
                    const std::function<bool(int row)>& isHeaderRow);

}} // namespace cv::docproc

#endif // OPENCV_DOCPROC_TEXT_UTILS_HPP
