// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DOCPROC_HPP
#define OPENCV_DOCPROC_HPP

#include "opencv2/core.hpp"
#include "opencv2/vlm.hpp"

/**
  @defgroup docproc Document Reading

  Convenience API built on top of the @ref vlm module: runs a VLM OCR / document-
  understanding engine over a document image and parses its raw output into a unified,
  engine-agnostic result -- pages made of typed blocks (title/text/table/form/signature),
  each carrying lines, words, table cells or form fields with bounding boxes and
  confidence scores. See docRead().
 */

namespace cv { namespace docproc {

//! @addtogroup docproc
//! @{

/** @brief Type of a parsed layout block within a page. */
enum BlockType
{
    BLOCK_TITLE     = 0,  //!< Page/section title.
    BLOCK_TEXT      = 1,  //!< Paragraph or other free-form text.
    BLOCK_TABLE     = 2,  //!< Tabular data; see Block::rows, Block::cols, Block::cells.
    BLOCK_FORM      = 3,  //!< Key/value form fields; see Block::fields.
    BLOCK_SIGNATURE = 4   //!< Signature / authorization block. No engine supported today
                          //!< reports one, so no parser produces this type yet.
};

/** @brief A single recognized word within a Line. */
class CV_EXPORTS_W_SIMPLE Word
{
public:
    CV_WRAP Word() : confidence(0.f) {}

    CV_PROP_RW String text;       //!< Recognized word text.
    CV_PROP_RW Rect bbox;         //!< Word bounding box, in page pixel coordinates. Empty
                                  //!< unless the engine reports word-level geometry; none
                                  //!< of the engines supported today do.
    CV_PROP_RW float confidence;  //!< Recognition confidence in [0, 1].
};

/** @brief A single recognized line of text within a Block, made up of Word%s. */
class CV_EXPORTS_W_SIMPLE Line
{
public:
    CV_WRAP Line() {}

    CV_PROP_RW String text;             //!< Full line text.
    CV_PROP_RW Rect bbox;               //!< Line bounding box, in page pixel coordinates.
                                        //!< Empty for engines whose output carries no
                                        //!< geometry; see docRead().
    CV_PROP_RW std::vector<Word> words; //!< Per-word breakdown; empty if the engine only
                                         //!< reports line-level text.
};

/** @brief A single cell within a Block of type BLOCK_TABLE. */
class CV_EXPORTS_W_SIMPLE Cell
{
public:
    CV_WRAP Cell() : row(0), col(0), is_header(false) {}

    CV_PROP_RW int row;         //!< Zero-based row index.
    CV_PROP_RW int col;         //!< Zero-based column index.
    CV_PROP_RW String text;     //!< Cell text; empty for blank cells.
    CV_PROP_RW bool is_header;  //!< True for header-row/column cells.
    CV_PROP_RW Rect bbox;       //!< Cell bounding box, in page pixel coordinates. Empty for
                                //!< engines whose output carries no geometry; see docRead().
};

/** @brief A single key/value pair within a Block of type BLOCK_FORM.

DocTags marks a key/value region but does not tag its keys and values separately, so each
child element of the region becomes one Field, split on its first ':'. A child with no ':'
becomes a #key with an empty #value.
*/
class CV_EXPORTS_W_SIMPLE Field
{
public:
    CV_WRAP Field() {}

    CV_PROP_RW String key;    //!< Field label, e.g. "Payment Terms".
    CV_PROP_RW String value;  //!< Field value, e.g. "Net 30".
    CV_PROP_RW Rect bbox;     //!< Field bounding box, in page pixel coordinates. Empty for
                              //!< engines whose output carries no geometry; see docRead().
};

/** @brief One parsed layout block on a page.

Only the members relevant to #type are populated: BLOCK_TITLE/BLOCK_TEXT/BLOCK_SIGNATURE
use #lines; BLOCK_TABLE uses #rows, #cols and #cells; BLOCK_FORM uses #fields.
*/
class CV_EXPORTS_W_SIMPLE Block
{
public:
    CV_WRAP Block() : type(BLOCK_TEXT), confidence(0.f), rows(0), cols(0) {}

    CV_PROP_RW BlockType type;    //!< Block type.
    CV_PROP_RW Rect bbox;         //!< Block bounding box, in page pixel coordinates: the union
                                  //!< of this block's lines' or cells' boxes. Empty for engines
                                  //!< whose output carries no geometry; see docRead().
    CV_PROP_RW float confidence;  //!< Block-level detection confidence in [0, 1].

    CV_PROP_RW std::vector<Line> lines;  //!< Populated for BLOCK_TITLE/BLOCK_TEXT/BLOCK_SIGNATURE.

    CV_PROP_RW int rows;                 //!< Row count; populated for BLOCK_TABLE.
    CV_PROP_RW int cols;                 //!< Column count; populated for BLOCK_TABLE.
    CV_PROP_RW std::vector<Cell> cells;  //!< Populated for BLOCK_TABLE.

    CV_PROP_RW std::vector<Field> fields; //!< Populated for BLOCK_FORM.
};

/** @brief One page of a parsed document. */
class CV_EXPORTS_W_SIMPLE Page
{
public:
    CV_WRAP Page() : page_number(0), width(0), height(0) {}

    CV_PROP_RW int page_number;            //!< One-based page number.
    CV_PROP_RW int width;                  //!< Page width, in pixels.
    CV_PROP_RW int height;                 //!< Page height, in pixels.
    CV_PROP_RW std::vector<Block> blocks;  //!< Layout blocks, in reading order.

    /** @brief Collects all recognized Word%s from this page's title/text/signature blocks,
    in reading order.

    @note Table cells and form fields carry no per-word breakdown (see Cell::text,
          Field::value), so those blocks are not searched.
    @return Words in reading order. For a Line whose Line::words the underlying engine left
            empty (line-level text only), Word%s are synthesized by splitting Line::text on
            whitespace; those synthesized Word%s carry text only (default Rect bbox, 0
            confidence).
    */
    CV_WRAP std::vector<Word> getWords() const;

    /** @brief Returns this page's BLOCK_TABLE blocks, in reading order. */
    CV_WRAP std::vector<Block> getTables() const;

    /** @brief Returns this page's title/text/signature lines, one Line::text per sentence,
    in reading order.

    @return Non-empty Line::text values; empty lines are skipped.
    */
    CV_WRAP std::vector<String> getSentences() const;
};

/** @brief Run/model metadata for a docRead() call. */
class CV_EXPORTS_W_SIMPLE DocumentMetadata
{
public:
    CV_WRAP DocumentMetadata() : pages(0), tokens_used(-1), inference_time_ms(0) {}

    CV_PROP_RW String model;    //!< Model identifier, e.g. "granite-docling-258m".
    CV_PROP_RW String engine;   //!< cv::dnn engine used, e.g. "ENGINE_NEW" or "ENGINE_ORT".
    CV_PROP_RW int pages;       //!< Number of pages processed.
    CV_PROP_RW int tokens_used; //!< Total tokens (prompt + completion), per
                                //!< vlm::VLMModel::lastTokensUsed(); -1 only if the underlying
                                //!< VLMModel doesn't override it.
    CV_PROP_RW int64 inference_time_ms;  //!< Total wall-clock inference time, in milliseconds.
};

/** @brief Unified result of a docRead() call.

If docRead() was called with raw=true, #pages is left empty and each page's unparsed
engine output is available instead in #raw_pages; #metadata is always populated.
*/
class CV_EXPORTS_W_SIMPLE DocumentResult
{
public:
    CV_WRAP DocumentResult() {}

    CV_PROP_RW DocumentMetadata metadata;      //!< Run/model metadata.
    CV_PROP_RW std::vector<Page> pages;        //!< Parsed pages; empty when raw=true was
                                                //!< passed to docRead().
    CV_PROP_RW std::vector<String> raw_pages;  //!< Unparsed per-page engine output; only
                                                //!< populated when raw=true was passed to
                                                //!< docRead().
};

//! @}

} // namespace docproc

//! @addtogroup docproc
//! @{

/** @brief Read a document image with a VLM engine and return a unified, structured result.

Wraps vlm::create() + vlm::VLMModel::inferDocument() and parses the engine's raw output
(OCR text, or DocTags-style markup, depending on model_type) into a page/block hierarchy --
see docproc::DocumentResult. Pass raw=true to skip parsing and get each page's unparsed
engine output instead, in docproc::DocumentResult::raw_pages.

@param input_path      Path to a `.png`/`.jpg`/`.jpeg` document image (treated as a single
                        page); see vlm::VLMModel::inferDocument().
@param model_type      Which VLM to load; see vlm::VLMModelType.
@param model_dir       For local model types: path to the local ONNX export directory.
                        For cloud model types: the provider's model name. See vlm::create().
@param engine          cv::dnn engine used for local model types. Only "opencv" is
                       supported, matching vlm::create(); ignored
                        by cloud model types. See vlm::create().
@param device          Compute device for local model types: "cpu" or "cuda"; ignored by
                        cloud model types. See vlm::create().
@param api_key         API key for cloud model types; ignored by local model types.
@param prompt          Task prompt passed to the engine; an empty string uses the
                        engine's default prompt.
@param max_new_tokens  Maximum number of tokens the engine may generate, per page.
@param raw             If true, skip structured parsing and return each page's raw engine
                        output in docproc::DocumentResult::raw_pages instead of
                        docproc::DocumentResult::pages.

@note Bounding boxes are only as good as the engine's output. DocTags-emitting engines
      (VLM_MODEL_GRANITE_DOCLING) localize each element, so docproc::Line::bbox,
      docproc::Cell::bbox and docproc::Block::bbox are filled. Engines that return plain OCR
      text or Markdown carry no geometry, so those stay empty. docproc::Word::bbox is always
      empty -- no supported engine reports word-level geometry.

@note In Python, when raw=True the call returns the list of per-page raw strings
      directly (equivalent to docproc::DocumentResult::raw_pages), not a DocumentResult.

@return Parsed (or, if raw=true, unparsed) document result; see docproc::DocumentResult.
*/
CV_EXPORTS_W docproc::DocumentResult docRead(CV_WRAP_FILE_PATH const String& input_path,
                                              vlm::VLMModelType model_type,
                                              CV_WRAP_FILE_PATH const String& model_dir,
                                              const String& engine = "opencv",
                                              const String& device = "cpu",
                                              const String& api_key = String(),
                                              const String& prompt = String(),
                                              int max_new_tokens = 512,
                                              bool raw = false);

//! @}

} // namespace cv

#endif // OPENCV_DOCPROC_HPP
