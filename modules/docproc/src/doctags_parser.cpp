// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "doctags_parser.hpp"
#include "text_utils.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <vector>

namespace cv { namespace docproc {

namespace {

struct Entry
{
    bool isTag;
    String name;     // valid when isTag; tag name without '<' '>' and without a leading '/'
    bool closing;     // valid when isTag; true for </name>
    String text;      // valid when !isTag; trimmed literal text run
};

std::vector<Entry> tokenize(const String& raw)
{
    std::vector<Entry> entries;
    size_t i = 0, n = raw.size();
    while (i < n)
    {
        size_t lt = raw.find('<', i);
        if (lt == String::npos)
        {
            String text = trimmed(raw.substr(i));
            if (!text.empty())
                entries.push_back({false, String(), false, text});
            break;
        }
        if (lt > i)
        {
            String text = trimmed(raw.substr(i, lt - i));
            if (!text.empty())
                entries.push_back({false, String(), false, text});
        }
        size_t gt = raw.find('>', lt);
        if (gt == String::npos)
            break;
        String inner = raw.substr(lt + 1, gt - lt - 1);
        bool closing = !inner.empty() && inner[0] == '/';
        entries.push_back({true, closing ? inner.substr(1) : inner, closing, String()});
        i = gt + 1;
    }
    return entries;
}

bool isLocTag(const String& name)
{
    return name.rfind("loc_", 0) == 0;
}

// <title> is the document title, <section_header_level_N> a section heading; both become
// BLOCK_TITLE.
bool isHeaderTag(const String& name)
{
    return name == "title" || name.rfind("section_header_level", 0) == 0;
}

// OTSL cell tags: f/e are filled/empty cells, l/u/x continue a span left/up/both, and
// ched/rhed are column and row header cells.
bool isTableCellTag(const String& name)
{
    return name == "fcel" || name == "ecel" || name == "lcel" || name == "ucel" ||
           name == "xcel" || name == "ched" || name == "rhed";
}

bool isHeaderCellTag(const String& name)
{
    return name == "ched" || name == "rhed";
}

bool isFormTag(const String& name)
{
    return name == "form" || name == "key_value_region";
}

// DocTags location tags are normalized to this range on both axes, independent of the
// page's pixel size.
const int DOCTAGS_LOC_RANGE = 500;

bool locValue(const String& name, int& value)
{
    if (!isLocTag(name) || name.size() <= 4)
        return false;
    for (size_t k = 4; k < name.size(); k++)
        if (!isdigit((unsigned char)name[k]))
            return false;
    value = atoi(name.c_str() + 4);
    return true;
}

// Reads the four <loc_N> tags an element carries between its opening tag and its content
// (<text><loc_x0><loc_y0><loc_x1><loc_y1>the text</text>) and scales them to pageSize.
// Returns false and leaves i untouched when they are absent -- not every element is
// localized, and bare OTSL cell streams carry no locations at all.
bool readLoc(const std::vector<Entry>& entries, size_t& i, Size pageSize, Rect& bbox)
{
    const size_t start = i;
    int v[4] = { 0, 0, 0, 0 };
    for (int k = 0; k < 4; k++)
    {
        if (i >= entries.size() || !entries[i].isTag || entries[i].closing ||
            !locValue(entries[i].name, v[k]))
        {
            i = start;
            return false;
        }
        i++;
    }

    const double sx = (double)pageSize.width / DOCTAGS_LOC_RANGE;
    const double sy = (double)pageSize.height / DOCTAGS_LOC_RANGE;
    const int x0 = cvRound(v[0] * sx), y0 = cvRound(v[1] * sy);
    const int x1 = cvRound(v[2] * sx), y1 = cvRound(v[3] * sy);
    bbox = Rect(x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0)) &
           Rect(Point(0, 0), pageSize);
    return true;
}

struct CellEntry
{
    String type;
    String text;
    Rect bbox;
};

void parseTable(const std::vector<Entry>& entries, size_t& i, Size pageSize, Block& block)
{
    std::vector<std::vector<CellEntry>> rows;
    std::vector<CellEntry> curRow;

    while (i < entries.size() &&
           !(entries[i].isTag && entries[i].closing && entries[i].name == "otsl"))
    {
        if (entries[i].isTag && isTableCellTag(entries[i].name))
        {
            CellEntry cell;
            cell.type = entries[i].name;
            i++;
            readLoc(entries, i, pageSize, cell.bbox);
            if (i < entries.size() && !entries[i].isTag)
            {
                cell.text = entries[i].text;
                i++;
            }
            curRow.push_back(cell);
        }
        else if (entries[i].isTag && entries[i].name == "nl")
        {
            rows.push_back(curRow);
            curRow.clear();
            i++;
        }
        else
        {
            i++;
        }
    }
    if (!curRow.empty())
        rows.push_back(curRow);
    if (i < entries.size())
        i++; // consume </otsl>

    block.rows = (int)rows.size();
    size_t maxCols = 0;
    for (const auto& row : rows)
        maxCols = std::max(maxCols, row.size());
    block.cols = (int)maxCols;

    for (int r = 0; r < (int)rows.size(); r++)
        for (int c = 0; c < (int)rows[r].size(); c++)
        {
            Cell cell;
            cell.row = r;
            cell.col = c;
            cell.text = rows[r][c].text;
            cell.is_header = isHeaderCellTag(rows[r][c].type);
            cell.bbox = rows[r][c].bbox;
            block.cells.push_back(cell);
            block.bbox |= cell.bbox;
        }
}

// Splits "Payment Terms: Net 30" on its first ':'; a child with no separator becomes a key
// with an empty value.
Field toField(const String& text, const Rect& bbox)
{
    Field field;
    field.bbox = bbox;
    const size_t colon = text.find(':');
    if (colon == String::npos)
    {
        field.key = text;
    }
    else
    {
        field.key = trimmed(text.substr(0, colon));
        field.value = trimmed(text.substr(colon + 1));
    }
    return field;
}

// DocTags marks a key/value region but does not tag its keys and values separately, so each
// child element becomes one Field, split on its first ':'. A checkbox child carries its label
// as the key and its state as the value.
void parseForm(const std::vector<Entry>& entries, size_t& i, const String& regionTag,
               Size pageSize, Block& block)
{
    while (i < entries.size() &&
           !(entries[i].isTag && entries[i].closing && entries[i].name == regionTag))
    {
        if (!entries[i].isTag)
        {
            block.fields.push_back(toField(entries[i].text, Rect()));
            i++;
            continue;
        }
        if (entries[i].closing)
        {
            i++;
            continue;
        }

        const String name = entries[i].name;
        i++;
        Rect bbox;
        readLoc(entries, i, pageSize, bbox);
        String text;
        if (i < entries.size() && !entries[i].isTag)
        {
            text = entries[i].text;
            i++;
        }
        if (i < entries.size() && entries[i].isTag && entries[i].closing &&
            entries[i].name == name)
            i++;

        if (text.empty())
            continue;

        Field field;
        if (name == "checkbox_selected" || name == "checkbox_unselected")
        {
            field.key = text;
            field.value = name == "checkbox_selected" ? "selected" : "unselected";
            field.bbox = bbox;
        }
        else
        {
            field = toField(text, bbox);
        }
        block.fields.push_back(field);
        block.bbox |= bbox;
    }
    if (i < entries.size())
        i++; // consume the region's closing tag
}

} // namespace

bool isDocTags(const String& raw)
{
    return raw.find("<doctag>") != String::npos;
}

bool isOtslCellStream(const String& raw)
{
    return raw.find("<fcel>") != String::npos || raw.find("<ched>") != String::npos;
}

void parseOtslCellStream(const String& raw, Page& page)
{
    std::vector<Entry> entries = tokenize(raw);
    size_t i = 0;
    Block block;
    block.type = BLOCK_TABLE;
    block.confidence = 1.f;
    // no </otsl> to stop at; runs to end of the stream
    parseTable(entries, i, Size(page.width, page.height), block);
    page.blocks.push_back(block);
}

void parseDocTagsPage(const String& raw, Page& page)
{
    std::vector<Entry> entries = tokenize(raw);
    size_t i = 0, n = entries.size();
    const Size pageSize(page.width, page.height);

    Block textAccum;
    bool haveTextAccum = false;

    auto flushTextAccum = [&]()
    {
        if (haveTextAccum && !textAccum.lines.empty())
            page.blocks.push_back(textAccum);
        textAccum = Block();
        textAccum.type = BLOCK_TEXT;
        textAccum.confidence = 1.f;
        haveTextAccum = false;
    };
    flushTextAccum();

    auto accumulateText = [&](const String& text, const Rect& bbox)
    {
        haveTextAccum = true;
        Line line;
        line.text = text;
        line.bbox = bbox;
        textAccum.lines.push_back(line);
        textAccum.bbox |= bbox;
    };

    while (i < n)
    {
        const Entry& e = entries[i];

        // Text that belongs to no element we consumed. Keep it: dropping it would lose page
        // content silently.
        if (!e.isTag)
        {
            accumulateText(e.text, Rect());
            i++;
            continue;
        }
        if (e.closing)
        {
            i++;
            continue;
        }

        const String name = e.name;
        if (name == "doctag")
        {
            i++;
            continue;
        }

        if (name == "otsl")
        {
            flushTextAccum();
            i++;
            Block block;
            block.type = BLOCK_TABLE;
            block.confidence = 1.f;
            readLoc(entries, i, pageSize, block.bbox);
            parseTable(entries, i, pageSize, block);
            page.blocks.push_back(block);
            continue;
        }

        if (isFormTag(name))
        {
            flushTextAccum();
            i++;
            Block block;
            block.type = BLOCK_FORM;
            block.confidence = 1.f;
            readLoc(entries, i, pageSize, block.bbox);
            parseForm(entries, i, name, pageSize, block);
            if (!block.fields.empty())
                page.blocks.push_back(block);
            continue;
        }

        // Every other element, including tags this parser does not know by name, may carry a
        // location and one text run. Wrapper tags such as <unordered_list> have no text of
        // their own, so they fall through here harmlessly and their children are handled on
        // the following iterations.
        i++;
        Rect bbox;
        readLoc(entries, i, pageSize, bbox);
        String text;
        if (i < n && !entries[i].isTag)
        {
            text = entries[i].text;
            i++;
        }
        if (i < n && entries[i].isTag && entries[i].closing && entries[i].name == name)
            i++;

        // <picture>/<chart> and other figures carry a location but no text.
        if (text.empty())
            continue;

        if (isHeaderTag(name))
        {
            flushTextAccum();
            Block block;
            block.type = BLOCK_TITLE;
            block.confidence = 1.f;
            block.bbox = bbox;
            Line line;
            line.text = text;
            line.bbox = bbox;
            block.lines.push_back(line);
            page.blocks.push_back(block);
        }
        else
        {
            accumulateText(text, bbox);
        }
    }

    flushTextAccum();
}

}} // namespace cv::docproc
