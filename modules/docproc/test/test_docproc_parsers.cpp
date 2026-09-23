// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "../src/doctags_parser.hpp"
#include "../src/markdown_parser.hpp"
#include "../src/plain_text_parser.hpp"

// The parsers are not exported from "opencv_docproc", so the only way to test them is to
// compile the source code into "opencv_test_docproc" (same approach as core's
// test_logtagmanager.cpp, and as the vlm module's test_vlm_internal.cpp).
#if 1
#include "../src/doctags_parser.cpp"
#include "../src/markdown_parser.cpp"
#include "../src/plain_text_parser.cpp"
#include "../src/text_utils.cpp"
#endif

namespace opencv_test { namespace {

using namespace cv::docproc;

TEST(Docproc_DocTagsParser, IsDocTags)
{
    EXPECT_TRUE(isDocTags("<doctag><text>hi</text></doctag>"));
    EXPECT_FALSE(isDocTags("plain text"));
}

TEST(Docproc_DocTagsParser, TextTagBecomesTextBlock)
{
    Page page;
    parseDocTagsPage("<doctag><text>Hello world</text></doctag>", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    EXPECT_EQ(BLOCK_TEXT, page.blocks[0].type);
    ASSERT_EQ((size_t)1, page.blocks[0].lines.size());
    EXPECT_EQ("Hello world", page.blocks[0].lines[0].text);
}

TEST(Docproc_DocTagsParser, HeaderTagBecomesTitleBlock)
{
    Page page;
    parseDocTagsPage(
        "<doctag><section_header_level_1>Title Text</section_header_level_1></doctag>", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    EXPECT_EQ(BLOCK_TITLE, page.blocks[0].type);
    EXPECT_EQ("Title Text", page.blocks[0].lines[0].text);
}

TEST(Docproc_DocTagsParser, TitleTagBecomesTitleBlock)
{
    Page page;
    parseDocTagsPage("<doctag><title>Annual Report</title></doctag>", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    EXPECT_EQ(BLOCK_TITLE, page.blocks[0].type);
    ASSERT_EQ((size_t)1, page.blocks[0].lines.size());
    EXPECT_EQ("Annual Report", page.blocks[0].lines[0].text);
}

TEST(Docproc_DocTagsParser, UnknownTagKeepsItsText)
{
    Page page;
    parseDocTagsPage("<doctag><paragraph>Body copy.</paragraph>"
                     "<footnote>A footnote.</footnote></doctag>", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    EXPECT_EQ(BLOCK_TEXT, page.blocks[0].type);
    ASSERT_EQ((size_t)2, page.blocks[0].lines.size());
    EXPECT_EQ("Body copy.", page.blocks[0].lines[0].text);
    EXPECT_EQ("A footnote.", page.blocks[0].lines[1].text);
}

TEST(Docproc_DocTagsParser, WrapperTagDoesNotSwallowItsChildren)
{
    Page page;
    parseDocTagsPage("<doctag><unordered_list><list_item>First</list_item>"
                     "<list_item>Second</list_item></unordered_list></doctag>", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    ASSERT_EQ((size_t)2, page.blocks[0].lines.size());
    EXPECT_EQ("First", page.blocks[0].lines[0].text);
    EXPECT_EQ("Second", page.blocks[0].lines[1].text);
}

TEST(Docproc_DocTagsParser, TextOutsideAnyElementIsKept)
{
    Page page;
    parseDocTagsPage("<doctag>Loose text with no element.</doctag>", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    ASSERT_EQ((size_t)1, page.blocks[0].lines.size());
    EXPECT_EQ("Loose text with no element.", page.blocks[0].lines[0].text);
}

TEST(Docproc_DocTagsParser, KeyValueRegionBecomesFormBlock)
{
    Page page;
    page.width = 500;
    page.height = 500;
    parseDocTagsPage("<doctag><key_value_region>"
                     "<text><loc_10><loc_10><loc_60><loc_30>Payment Terms: Net 30</text>"
                     "<text>Invoice ID: INV-2048</text>"
                     "</key_value_region></doctag>", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    const Block& block = page.blocks[0];
    EXPECT_EQ(BLOCK_FORM, block.type);
    ASSERT_EQ((size_t)2, block.fields.size());
    EXPECT_EQ("Payment Terms", block.fields[0].key);
    EXPECT_EQ("Net 30", block.fields[0].value);
    EXPECT_EQ(Rect(10, 10, 50, 20), block.fields[0].bbox);
    EXPECT_EQ("Invoice ID", block.fields[1].key);
    EXPECT_EQ("INV-2048", block.fields[1].value);
    EXPECT_TRUE(block.fields[1].bbox.empty());
    EXPECT_TRUE(block.lines.empty());
}

TEST(Docproc_DocTagsParser, FormFieldWithoutSeparatorKeepsWholeTextAsKey)
{
    Page page;
    parseDocTagsPage("<doctag><form><text>Signed and agreed</text></form></doctag>", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    ASSERT_EQ((size_t)1, page.blocks[0].fields.size());
    EXPECT_EQ("Signed and agreed", page.blocks[0].fields[0].key);
    EXPECT_TRUE(page.blocks[0].fields[0].value.empty());
}

TEST(Docproc_DocTagsParser, CheckboxChildCarriesItsState)
{
    Page page;
    parseDocTagsPage("<doctag><form>"
                     "<checkbox_selected>Expedited shipping</checkbox_selected>"
                     "<checkbox_unselected>Gift wrap</checkbox_unselected>"
                     "</form></doctag>", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    ASSERT_EQ((size_t)2, page.blocks[0].fields.size());
    EXPECT_EQ("Expedited shipping", page.blocks[0].fields[0].key);
    EXPECT_EQ("selected", page.blocks[0].fields[0].value);
    EXPECT_EQ("Gift wrap", page.blocks[0].fields[1].key);
    EXPECT_EQ("unselected", page.blocks[0].fields[1].value);
}

TEST(Docproc_DocTagsParser, RowHeaderAndSpanCellsAreRecognized)
{
    Page page;
    parseDocTagsPage("<doctag><otsl><rhed>Region<fcel>Q1<ucel><xcel><nl></otsl></doctag>", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    const Block& block = page.blocks[0];
    EXPECT_EQ(1, block.rows);
    EXPECT_EQ(4, block.cols); // rhed/ucel/xcel count as cells, not unknown tags
    ASSERT_EQ((size_t)4, block.cells.size());
    EXPECT_TRUE(block.cells[0].is_header); // <rhed>
    EXPECT_EQ("Region", block.cells[0].text);
    EXPECT_FALSE(block.cells[1].is_header);
}

TEST(Docproc_DocTagsParser, RaggedTableColsMatchesWidestRow)
{
    Page page;
    parseDocTagsPage(
        "<doctag><otsl><fcel>A<fcel>B<nl><fcel>C<fcel>D<fcel>E<nl></otsl></doctag>", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    const Block& block = page.blocks[0];
    EXPECT_EQ(BLOCK_TABLE, block.type);
    EXPECT_EQ(2, block.rows);
    EXPECT_EQ(3, block.cols); // widest row (3 cells), not the first row (2 cells)
    ASSERT_EQ((size_t)5, block.cells.size());
}

TEST(Docproc_DocTagsParser, LocTagsBecomeBoundingBoxes)
{
    Page page;
    page.width = 500;  // 1:1 with the DocTags 0..500 location range
    page.height = 500;
    parseDocTagsPage("<doctag><text><loc_10><loc_20><loc_110><loc_70>Hello</text></doctag>",
                     page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    ASSERT_EQ((size_t)1, page.blocks[0].lines.size());
    EXPECT_EQ(Rect(10, 20, 100, 50), page.blocks[0].lines[0].bbox);
    EXPECT_EQ(Rect(10, 20, 100, 50), page.blocks[0].bbox); // union of a single line
}

TEST(Docproc_DocTagsParser, LocTagsScaleToPageSize)
{
    Page page;
    page.width = 1000;  // 2x the location range on both axes
    page.height = 1000;
    parseDocTagsPage("<doctag><text><loc_10><loc_20><loc_110><loc_70>Hello</text></doctag>",
                     page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    ASSERT_EQ((size_t)1, page.blocks[0].lines.size());
    EXPECT_EQ(Rect(20, 40, 200, 100), page.blocks[0].lines[0].bbox);
}

TEST(Docproc_DocTagsParser, BlockBboxIsUnionOfItsLines)
{
    Page page;
    page.width = 500;
    page.height = 500;
    parseDocTagsPage("<doctag>"
                     "<text><loc_10><loc_20><loc_60><loc_40>First</text>"
                     "<text><loc_30><loc_50><loc_110><loc_70>Second</text>"
                     "</doctag>", page);

    ASSERT_EQ((size_t)1, page.blocks.size()); // both accumulate into one BLOCK_TEXT
    ASSERT_EQ((size_t)2, page.blocks[0].lines.size());
    EXPECT_EQ(Rect(10, 20, 100, 50), page.blocks[0].bbox);
}

TEST(Docproc_DocTagsParser, PictureWithoutTextContributesNoLine)
{
    Page page;
    page.width = 500;
    page.height = 500;
    parseDocTagsPage("<doctag>"
                     "<text><loc_10><loc_20><loc_60><loc_40>Caption</text>"
                     "<picture><loc_100><loc_100><loc_200><loc_200></picture>"
                     "</doctag>", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    ASSERT_EQ((size_t)1, page.blocks[0].lines.size()); // no empty line for <picture>
    EXPECT_EQ("Caption", page.blocks[0].lines[0].text);
}

TEST(Docproc_DocTagsParser, UnlocalizedElementsKeepAnEmptyBbox)
{
    Page page;
    page.width = 500;
    page.height = 500;
    parseDocTagsPage("<doctag><text>No location tags here</text></doctag>", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    ASSERT_EQ((size_t)1, page.blocks[0].lines.size());
    EXPECT_EQ("No location tags here", page.blocks[0].lines[0].text);
    EXPECT_TRUE(page.blocks[0].lines[0].bbox.empty());
}

TEST(Docproc_DocTagsParser, TableCellsCarryBoundingBoxes)
{
    Page page;
    page.width = 500;
    page.height = 500;
    parseDocTagsPage("<doctag><otsl>"
                     "<ched><loc_10><loc_10><loc_60><loc_30>A"
                     "<fcel><loc_60><loc_10><loc_110><loc_30>B<nl>"
                     "</otsl></doctag>", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    const Block& block = page.blocks[0];
    ASSERT_EQ((size_t)2, block.cells.size());
    EXPECT_EQ(Rect(10, 10, 50, 20), block.cells[0].bbox);
    EXPECT_EQ(Rect(60, 10, 50, 20), block.cells[1].bbox);
    EXPECT_EQ(Rect(10, 10, 100, 20), block.bbox); // union of both cells
    EXPECT_TRUE(block.cells[0].is_header);
    EXPECT_FALSE(block.cells[1].is_header);
}

TEST(Docproc_DocTagsParser, IsOtslCellStream)
{
    EXPECT_TRUE(isOtslCellStream("<fcel>A<fcel>B<nl>"));
    EXPECT_FALSE(isOtslCellStream("<doctag><text>hi</text></doctag>"));
}

TEST(Docproc_DocTagsParser, ParseOtslCellStreamHasNoClosingTag)
{
    Page page;
    parseOtslCellStream("<fcel>A<fcel>B<nl>", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    const Block& block = page.blocks[0];
    EXPECT_EQ(BLOCK_TABLE, block.type);
    EXPECT_EQ(1, block.rows);
    EXPECT_EQ(2, block.cols);
    ASSERT_EQ((size_t)2, block.cells.size());
    EXPECT_EQ("A", block.cells[0].text);
    EXPECT_EQ("B", block.cells[1].text);
}

TEST(Docproc_MarkdownParser, IsMarkdown)
{
    EXPECT_TRUE(isMarkdown("# Title\n"));
    EXPECT_TRUE(isMarkdown("| A | B |\n| --- | --- |\n| 1 | 2 |\n"));
    EXPECT_FALSE(isMarkdown("Just plain text.\nAnother line.\n"));
}

TEST(Docproc_MarkdownParser, HeaderAndParagraph)
{
    Page page;
    parseMarkdownPage("# Title\n\nSome paragraph text.", page);

    ASSERT_EQ((size_t)2, page.blocks.size());
    EXPECT_EQ(BLOCK_TITLE, page.blocks[0].type);
    EXPECT_EQ("Title", page.blocks[0].lines[0].text);
    EXPECT_EQ(BLOCK_TEXT, page.blocks[1].type);
    EXPECT_EQ("Some paragraph text.", page.blocks[1].lines[0].text);
}

TEST(Docproc_MarkdownParser, PipeTable)
{
    Page page;
    parseMarkdownPage("| A | B |\n| --- | --- |\n| 1 | 2 |", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    const Block& block = page.blocks[0];
    EXPECT_EQ(BLOCK_TABLE, block.type);
    EXPECT_EQ(2, block.rows);
    EXPECT_EQ(2, block.cols);
    ASSERT_EQ((size_t)4, block.cells.size());
    EXPECT_TRUE(block.cells[0].is_header);
    EXPECT_FALSE(block.cells[2].is_header);
    EXPECT_EQ("1", block.cells[2].text);
}

TEST(Docproc_PlainTextParser, SingleSpaceStaysInOneColumn)
{
    Page page;
    parsePlainTextPage("Jane Miller works here.", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    EXPECT_EQ(BLOCK_TEXT, page.blocks[0].type);
    ASSERT_EQ((size_t)1, page.blocks[0].lines.size());
    EXPECT_EQ("Jane Miller works here.", page.blocks[0].lines[0].text);
}

TEST(Docproc_PlainTextParser, WhitespaceAlignedColumnsBecomeTable)
{
    Page page;
    parsePlainTextPage("Name         Age\nJane Miller  30\nJohn Smith   25", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    const Block& block = page.blocks[0];
    EXPECT_EQ(BLOCK_TABLE, block.type);
    EXPECT_EQ(3, block.rows);
    EXPECT_EQ(2, block.cols);
    ASSERT_EQ((size_t)6, block.cells.size());
    EXPECT_EQ("Jane Miller", block.cells[2].text); // row 1, col 0 -- not split on the single space
    EXPECT_FALSE(block.cells[2].is_header);
}

TEST(Docproc_Page, GetWordsCollectsFromTextBearingBlocksOnly)
{
    Page page;

    Word w1; w1.text = "Hello";
    Line titleLine; titleLine.words = {w1};
    Block title; title.type = BLOCK_TITLE; title.lines = {titleLine};

    Word w2; w2.text = "Some";
    Word w3; w3.text = "text";
    Line textLine; textLine.words = {w2, w3};
    Block text; text.type = BLOCK_TEXT; text.lines = {textLine};

    Cell cell; cell.text = "1"; // table cells carry no Word breakdown
    Block table; table.type = BLOCK_TABLE; table.cells = {cell};

    page.blocks = {title, text, table};

    std::vector<Word> words = page.getWords();
    ASSERT_EQ((size_t)3, words.size());
    EXPECT_EQ("Hello", words[0].text);
    EXPECT_EQ("Some", words[1].text);
    EXPECT_EQ("text", words[2].text);
}

TEST(Docproc_Page, GetWordsSynthesizesFromLineTextWhenWordsEmpty)
{
    Page page;
    Line line; line.text = "Jane Miller works here.";
    Block text; text.type = BLOCK_TEXT; text.lines = {line};
    page.blocks = {text};

    std::vector<Word> words = page.getWords();
    ASSERT_EQ((size_t)4, words.size());
    EXPECT_EQ("Jane", words[0].text);
    EXPECT_EQ("Miller", words[1].text);
    EXPECT_EQ("works", words[2].text);
    EXPECT_EQ("here.", words[3].text);
}

TEST(Docproc_Page, GetTablesReturnsOnlyTableBlocksInOrder)
{
    Page page;
    Block text; text.type = BLOCK_TEXT;
    Block table1; table1.type = BLOCK_TABLE; table1.rows = 1; table1.cols = 1;
    Block form; form.type = BLOCK_FORM;
    Block table2; table2.type = BLOCK_TABLE; table2.rows = 2; table2.cols = 2;
    page.blocks = {text, table1, form, table2};

    std::vector<Block> tables = page.getTables();
    ASSERT_EQ((size_t)2, tables.size());
    EXPECT_EQ(1, tables[0].rows);
    EXPECT_EQ(2, tables[1].rows);
}

TEST(Docproc_Page, GetSentencesReturnsOneSentencePerLine)
{
    Page page;
    Line line1; line1.text = "Name: Jane Miller";
    Line line2; line2.text = "Invoice ID: INV-2048";
    Block block; block.type = BLOCK_TEXT; block.lines = {line1, line2};
    page.blocks = {block};

    std::vector<String> sentences = page.getSentences();
    ASSERT_EQ((size_t)2, sentences.size());
    EXPECT_EQ("Name: Jane Miller", sentences[0]);
    EXPECT_EQ("Invoice ID: INV-2048", sentences[1]);
}

TEST(Docproc_Page, GetSentencesSkipsEmptyLines)
{
    Page page;
    Line line1; line1.text = "First.";
    Line line2; line2.text = "";
    Line line3; line3.text = "Second.";
    Block block; block.type = BLOCK_TEXT; block.lines = {line1, line2, line3};
    page.blocks = {block};

    std::vector<String> sentences = page.getSentences();
    ASSERT_EQ((size_t)2, sentences.size());
    EXPECT_EQ("First.", sentences[0]);
    EXPECT_EQ("Second.", sentences[1]);
}

TEST(Docproc_Page, GetSentencesSkipsNonTextBearingBlocks)
{
    Page page;
    Line line; line.text = "Real sentence.";
    Block text; text.type = BLOCK_TEXT; text.lines = {line};

    Cell cell; cell.text = "Not a sentence. Should not appear.";
    Block table; table.type = BLOCK_TABLE; table.cells = {cell};

    page.blocks = {text, table};

    std::vector<String> sentences = page.getSentences();
    ASSERT_EQ((size_t)1, sentences.size());
    EXPECT_EQ("Real sentence.", sentences[0]);
}

}} // namespace opencv_test::(anonymous)
