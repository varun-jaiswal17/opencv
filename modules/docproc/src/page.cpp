// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"

#include <cctype>

namespace cv { namespace docproc {

namespace {

bool isTextBearing(BlockType type)
{
    return type == BLOCK_TITLE || type == BLOCK_TEXT || type == BLOCK_SIGNATURE;
}

std::vector<Word> synthesizeWords(const String& text)
{
    std::vector<Word> words;
    size_t i = 0, n = text.size();
    while (i < n)
    {
        while (i < n && isspace((unsigned char)text[i]))
            i++;
        size_t start = i;
        while (i < n && !isspace((unsigned char)text[i]))
            i++;
        if (i > start)
        {
            Word word;
            word.text = text.substr(start, i - start);
            words.push_back(word);
        }
    }
    return words;
}

} // namespace

std::vector<Word> Page::getWords() const
{
    std::vector<Word> words;
    for (const Block& block : blocks)
    {
        if (!isTextBearing(block.type))
            continue;
        for (const Line& line : block.lines)
        {
            if (!line.words.empty())
                words.insert(words.end(), line.words.begin(), line.words.end());
            else
            {
                std::vector<Word> synthesized = synthesizeWords(line.text);
                words.insert(words.end(), synthesized.begin(), synthesized.end());
            }
        }
    }
    return words;
}

std::vector<Block> Page::getTables() const
{
    std::vector<Block> tables;
    for (const Block& block : blocks)
        if (block.type == BLOCK_TABLE)
            tables.push_back(block);
    return tables;
}

std::vector<String> Page::getSentences() const
{
    std::vector<String> sentences;
    for (const Block& block : blocks)
    {
        if (!isTextBearing(block.type))
            continue;
        for (const Line& line : block.lines)
            if (!line.text.empty())
                sentences.push_back(line.text);
    }
    return sentences;
}

}} // namespace cv::docproc
