// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "plain_text_parser.hpp"
#include "text_utils.hpp"

#include <vector>

namespace cv { namespace docproc {

namespace {

bool isSpaceOrTab(char c) { return c == ' ' || c == '\t'; }

// Splits a line into columns on runs of 2+ spaces/tabs; a single space stays part of a column
// (so e.g. "Jane Miller" is one column, not two).
std::vector<String> splitColumns(const String& line)
{
    std::vector<String> cols;
    size_t i = 0, n = line.size();
    while (i < n)
    {
        size_t tokenStart = i;
        while (i < n)
        {
            if (isSpaceOrTab(line[i]))
            {
                size_t j = i;
                while (j < n && isSpaceOrTab(line[j]))
                    j++;
                if (j - i >= 2)
                    break;
                i = j;
            }
            else
            {
                i++;
            }
        }
        cols.push_back(trimmed(line.substr(tokenStart, i - tokenStart)));
        while (i < n && isSpaceOrTab(line[i]))
            i++;
    }
    return cols;
}

} // namespace

void parsePlainTextPage(const String& raw, Page& page)
{
    std::vector<String> lines = splitLines(raw);
    size_t i = 0, n = lines.size();

    std::vector<Line> textAccum;
    auto flushText = [&]()
    {
        if (!textAccum.empty())
            page.blocks.push_back(makeTextBlock(textAccum));
    };

    while (i < n)
    {
        if (lines[i].empty())
        {
            flushText();
            i++;
            continue;
        }

        std::vector<String> cols = splitColumns(lines[i]);
        if (cols.size() >= 2)
        {
            size_t j = i;
            std::vector<std::vector<String>> rows;
            while (j < n && !lines[j].empty())
            {
                std::vector<String> rowCols = splitColumns(lines[j]);
                if (rowCols.size() != cols.size())
                    break;
                rows.push_back(rowCols);
                j++;
            }
            if (rows.size() >= 2)
            {
                flushText();
                Block block;
                block.type = BLOCK_TABLE;
                block.confidence = 1.f;
                block.rows = (int)rows.size();
                block.cols = (int)cols.size();
                fillTableCells(block, rows, [](int r) { return r == 0; });
                page.blocks.push_back(block);
                i = j;
                continue;
            }
        }

        Line line;
        line.text = lines[i];
        textAccum.push_back(line);
        i++;
    }
    flushText();
}

}} // namespace cv::docproc
