// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_SRC_MAPPED_FILE_HPP
#define OPENCV_DNN_SRC_MAPPED_FILE_HPP

#include <string>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

// A read-only mapping of a weight file: a constant nothing rewrites becomes a Mat header over
// it instead of a heap copy. Read-only so the view carries no commit charge.
class MappedFile
{
public:
    ~MappedFile();

    // Empty Ptr when the platform or the file cannot supply a mapping; callers copy instead.
    static Ptr<MappedFile> open(const std::string& path);

    // Mat over this mapping that holds a reference to it, so the Mat outlives the Net like
    // every other cv::Mat its caller is handed.
    static Mat wrap(const Ptr<MappedFile>& file, int dims, const int* sizes, int type,
                    uchar* data);

    uchar* data() const { return base; }
    size_t size() const { return length; }

private:
    MappedFile() {}
    MappedFile(const MappedFile&);
    MappedFile& operator=(const MappedFile&);

    uchar* base = nullptr;
    size_t length = 0;
};

CV__DNN_INLINE_NS_END
}}

#endif
