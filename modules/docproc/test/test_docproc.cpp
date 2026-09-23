// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "opencv2/core/utils/configuration.private.hpp"
#include "opencv2/core/utils/filesystem.hpp"

namespace opencv_test { namespace {

using namespace cv::docproc;
using cv::docRead;

class Docproc_DocRead : public testing::Test
{
protected:
    void SetUp() override
    {
        imagePath = cv::tempfile(".png");
        cv::imwrite(imagePath, cv::Mat::zeros(32, 32, CV_8UC3));
    }

    void TearDown() override
    {
        cv::utils::fs::remove_all(imagePath);
    }

    std::string imagePath;
};

TEST_F(Docproc_DocRead, NonexistentModelDirThrows)
{
    EXPECT_THROW(docRead(imagePath, cv::vlm::VLM_MODEL_PADDLEOCR_VL, "/nonexistent/model/dir"),
                 cv::Exception);
}

TEST_F(Docproc_DocRead, EndToEnd_PaddleOCRVL)
{
    std::string modelDir =
        cv::utils::getConfigurationParameterString("OPENCV_TEST_VLM_PADDLEOCR_VL_DIR");
    if (modelDir.empty())
        throw SkipTestException(
            "OPENCV_TEST_VLM_PADDLEOCR_VL_DIR is not set; skipping end-to-end test");

    std::string testImagePath = cv::utils::getConfigurationParameterString("OPENCV_TEST_VLM_IMAGE");
    ASSERT_FALSE(testImagePath.empty())
        << "OPENCV_TEST_VLM_IMAGE must be set together with OPENCV_TEST_VLM_PADDLEOCR_VL_DIR";

    DocumentResult structured = docRead(testImagePath, cv::vlm::VLM_MODEL_PADDLEOCR_VL, modelDir);
    EXPECT_EQ("paddleocr-vl-1.5", structured.metadata.model);
    ASSERT_EQ((size_t)1, structured.pages.size());
    EXPECT_TRUE(structured.raw_pages.empty());
    EXPECT_GT(structured.pages[0].blocks.size(), (size_t)0);

    DocumentResult raw = docRead(testImagePath, cv::vlm::VLM_MODEL_PADDLEOCR_VL, modelDir,
                                  "opencv", "cpu", "", "", 512, true);
    EXPECT_TRUE(raw.pages.empty());
    ASSERT_EQ((size_t)1, raw.raw_pages.size());
    EXPECT_FALSE(raw.raw_pages[0].empty());
}

}} // namespace
