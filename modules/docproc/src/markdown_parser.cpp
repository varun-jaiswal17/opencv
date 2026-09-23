// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "markdown_parser.hpp"
#include "text_utils.hpp"

#include <vector>

namespace cv { namespace docproc {

namespace {

bool isHeaderLine(const String& line, String& text)
{
    size_t i = 0;
    while (i < line.size() && line[i] == '#')
        i++;
    if (i == 0 || i >= line.size() || line[i] != ' ')
        return false;
    text = trimmed(line.substr(i + 1));
    return true;
}

std::vector<String> splitTableRow(const String& line)
{
    String t = line;
    if (!t.empty() && t.front() == '|')
        t = t.substr(1);
    if (!t.empty() && t.back() == '|')
        t = t.substr(0, t.size() - 1);

    std::vector<String> cells;
    size_t start = 0;
    for (size_t i = 0; i <= t.size(); i++)
    {
        if (i == t.size() || t[i] == '|')
        {
            cells.push_back(trimmed(t.substr(start, i - start)));
            start = i + 1;
        }
    }
    return cells;
}

bool isSeparatorCell(const String& cell)
{
    if (cell.empty())
        return false;
    bool hasDash = false;
    for (char c : cell)
    {
        if (c == '-')
            hasDash = true;
        else if (c != ':')
            return false;
    }
    return hasDash;
}

bool isSeparatorRow(const std::vector<String>& cells)
{
    if (cells.empty())
        return false;
    for (const String& c : cells)
        if (!isSeparatorCell(c))
            return false;
    return true;
}

} // namespace

bool isMarkdown(const String& raw)
{
    std::vector<String> lines = splitLines(raw);
    for (size_t i = 0; i < lines.size(); i++)
    {
        String headerText;
        if (isHeaderLine(lines[i], headerText))
            return true;
        if (i + 1 < lines.size())
        {
            std::vector<String> headerCells = splitTableRow(lines[i]);
            std::vector<String> sepCells = splitTableRow(lines[i + 1]);
            if (headerCells.size() >= 2 && sepCells.size() == headerCells.size() &&
                isSeparatorRow(sepCells))
                return true;
        }
    }
    return false;
}

void parseMarkdownPage(const String& raw, Page& page)
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

        String headerText;
        if (isHeaderLine(lines[i], headerText))
        {
            flushText();
            Block block;
            block.type = BLOCK_TITLE;
            block.confidence = 1.f;
            Line line;
            line.text = headerText;
            block.lines.push_back(line);
            page.blocks.push_back(block);
            i++;
            continue;
        }

        std::vector<String> headerCells = splitTableRow(lines[i]);
        if (headerCells.size() >= 2 && i + 1 < n)
        {
            std::vector<String> sepCells = splitTableRow(lines[i + 1]);
            if (sepCells.size() == headerCells.size() && isSeparatorRow(sepCells))
            {
                flushText();
                std::vector<std::vector<String>> rows;
                rows.push_back(headerCells);
                size_t j = i + 2;
                while (j < n && !lines[j].empty())
                {
                    std::vector<String> rowCells = splitTableRow(lines[j]);
                    if (rowCells.size() != headerCells.size())
                        break;
                    rows.push_back(rowCells);
                    j++;
                }

                Block block;
                block.type = BLOCK_TABLE;
                block.confidence = 1.f;
                block.rows = (int)rows.size();
                block.cols = (int)headerCells.size();
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
