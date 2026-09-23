// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"
#include "../src/doctags_parser.hpp"
#include "../src/markdown_parser.hpp"
#include "../src/plain_text_parser.hpp"

// The parsers are not exported from "opencv_docproc", so the only way to measure them is
// to compile the source code into "opencv_perf_docproc" (same approach as core's
// test_logtagmanager.cpp, and as the vlm module's perf_vlm_model.cpp).
#if 1
#include "../src/doctags_parser.cpp"
#include "../src/markdown_parser.cpp"
#include "../src/plain_text_parser.cpp"
#include "../src/text_utils.cpp"
#endif

#include <string>

namespace opencv_test {

static String buildDocTagsPage(int numParagraphs, int tableRows, int tableCols)
{
    String raw = "<doctag><section_header_level_1>Report</section_header_level_1>";
    for (int i = 0; i < numParagraphs; i++)
        raw += "<text>Paragraph " + std::to_string(i) + " of the synthetic report body.</text>";

    raw += "<otsl>";
    for (int r = 0; r < tableRows; r++)
    {
        for (int c = 0; c < tableCols; c++)
        {
            raw += (r == 0 ? "<ched>" : "<fcel>");
            raw += "R" + std::to_string(r) + "C" + std::to_string(c);
        }
        raw += "<nl>";
    }
    raw += "</otsl></doctag>";
    return raw;
}

static String buildMarkdownPage(int numParagraphs, int tableRows, int tableCols)
{
    String raw = "# Report\n\n";
    for (int i = 0; i < numParagraphs; i++)
        raw += "Paragraph " + std::to_string(i) + " of the synthetic report body.\n\n";

    String header, sep;
    for (int c = 0; c < tableCols; c++)
    {
        header += (c ? " | " : "") + ("Col" + std::to_string(c));
        sep += (c ? " | " : "") + String("---");
    }
    raw += "| " + header + " |\n| " + sep + " |\n";
    for (int r = 0; r < tableRows; r++)
    {
        String row;
        for (int c = 0; c < tableCols; c++)
            row += (c ? " | " : "") + ("R" + std::to_string(r) + "C" + std::to_string(c));
        raw += "| " + row + " |\n";
    }
    return raw;
}

static String buildPlainTextPage(int numParagraphs, int tableRows, int tableCols)
{
    String raw;
    for (int i = 0; i < numParagraphs; i++)
        raw += "Paragraph " + std::to_string(i) + " of the synthetic report body.\n\n";

    for (int r = 0; r < tableRows; r++)
    {
        String row;
        for (int c = 0; c < tableCols; c++)
            row += (c ? "  " : "") + ("R" + std::to_string(r) + "C" + std::to_string(c));
        raw += row + "\n";
    }
    return raw;
}

PERF_TEST(Docproc_DocTagsParser, ParsePage)
{
    String raw = buildDocTagsPage(200, 50, 6);
    Page page;
    TEST_CYCLE()
    {
        page = Page();
        parseDocTagsPage(raw, page);
    }
    SANITY_CHECK_NOTHING();
}

PERF_TEST(Docproc_MarkdownParser, ParsePage)
{
    String raw = buildMarkdownPage(200, 50, 6);
    Page page;
    TEST_CYCLE()
    {
        page = Page();
        parseMarkdownPage(raw, page);
    }
    SANITY_CHECK_NOTHING();
}

PERF_TEST(Docproc_PlainTextParser, ParsePage)
{
    String raw = buildPlainTextPage(200, 50, 6);
    Page page;
    TEST_CYCLE()
    {
        page = Page();
        parsePlainTextPage(raw, page);
    }
    SANITY_CHECK_NOTHING();
}

} // namespace opencv_test
