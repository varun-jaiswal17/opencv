// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "text_utils.hpp"

namespace cv { namespace docproc {

String trimmed(const String& s)
{
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == String::npos)
        return String();
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

std::vector<String> splitLines(const String& raw)
{
    std::vector<String> lines;
    size_t start = 0;
    for (size_t i = 0; i <= raw.size(); i++)
    {
        if (i == raw.size() || raw[i] == '\n')
        {
            lines.push_back(trimmed(raw.substr(start, i - start)));
            start = i + 1;
        }
    }
    return lines;
}

Block makeTextBlock(std::vector<Line>& lines)
{
    Block block;
    block.type = BLOCK_TEXT;
    block.confidence = 1.f;
    block.lines.swap(lines);
    return block;
}

void fillTableCells(Block& block, const std::vector<std::vector<String>>& rows,
                    const std::function<bool(int row)>& isHeaderRow)
{
    for (int r = 0; r < (int)rows.size(); r++)
        for (int c = 0; c < (int)rows[r].size(); c++)
        {
            Cell cell;
            cell.row = r;
            cell.col = c;
            cell.text = rows[r][c];
            cell.is_header = isHeaderRow(r);
            block.cells.push_back(cell);
        }
}

}} // namespace cv::docproc
