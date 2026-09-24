// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "mapped_file.hpp"

#ifdef _WIN32
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#define DNN_MMAP_AVAILABLE 1
#endif

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

namespace {

// Holds the mapping in UMatData::userdata so it is released with the last Mat over it, the
// way NumpyAllocator (modules/python/src2) keeps a PyObject alive behind a wrapped array.
class MappedFileAllocator CV_FINAL : public MatAllocator
{
public:
    MappedFileAllocator() : stdAllocator(Mat::getStdAllocator()) {}

    UMatData* allocate(int dims, const int* sizes, int type, void* data, size_t* step,
                       AccessFlag flags, UMatUsageFlags usageFlags) const CV_OVERRIDE
    {
        return stdAllocator->allocate(dims, sizes, type, data, step, flags, usageFlags);
    }

    bool allocate(UMatData* u, AccessFlag accessFlags, UMatUsageFlags usageFlags) const CV_OVERRIDE
    {
        return stdAllocator->allocate(u, accessFlags, usageFlags);
    }

    void deallocate(UMatData* u) const CV_OVERRIDE
    {
        if (!u)
            return;
        if (u->refcount == 0)
        {
            delete static_cast<Ptr<MappedFile>*>(u->userdata);
            delete u;
        }
    }

    const MatAllocator* stdAllocator;
};

static MappedFileAllocator& mappedFileAllocator()
{
    static MappedFileAllocator allocator;
    return allocator;
}

}

Mat MappedFile::wrap(const Ptr<MappedFile>& file, int dims, const int* sizes, int type,
                     uchar* data)
{
    Mat m(dims, sizes, type, data);
    UMatData* u = new UMatData(&mappedFileAllocator());
    u->data = u->origdata = data;
    u->size = m.total() * m.elemSize();
    u->userdata = new Ptr<MappedFile>(file);
    m.u = u;
    m.addref();
    m.allocator = &mappedFileAllocator();
    return m;
}

MappedFile::~MappedFile()
{
    if (!base)
        return;
#ifdef _WIN32
    UnmapViewOfFile(base);
#elif defined(DNN_MMAP_AVAILABLE)
    munmap(base, length);
#endif
}

Ptr<MappedFile> MappedFile::open(const std::string& path)
{
#ifdef _WIN32
    const int len = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                        path.c_str(), (int)path.size(), NULL, 0);
    if (len <= 0)
        return Ptr<MappedFile>();
    std::wstring widePath((size_t)len, L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                        path.c_str(), (int)path.size(), &widePath[0], len);

    HANDLE f = CreateFileW(widePath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (f == INVALID_HANDLE_VALUE)
        return Ptr<MappedFile>();
    LARGE_INTEGER sz;
    HANDLE m = NULL;
    void* p = NULL;
    if (GetFileSizeEx(f, &sz) && sz.QuadPart > 0)
        m = CreateFileMappingW(f, NULL, PAGE_READONLY, 0, 0, NULL);
    // The view holds the section and the section the file, so neither handle outlives this
    // call and the model file is not kept open for the Net's life.
    if (m)
    {
        p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(m);
    }
    CloseHandle(f);
    if (!p)
        return Ptr<MappedFile>();
    Ptr<MappedFile> mapped(new MappedFile());
    mapped->base = (uchar*)p;
    mapped->length = (size_t)sz.QuadPart;
    return mapped;
#elif defined(DNN_MMAP_AVAILABLE)
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
        return Ptr<MappedFile>();
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0)
    {
        ::close(fd);
        return Ptr<MappedFile>();
    }
    size_t len = (size_t)st.st_size;
    void* p = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
    // The mapping keeps its own reference to the file, so the descriptor is done here.
    ::close(fd);
    if (p == MAP_FAILED)
        return Ptr<MappedFile>();
    Ptr<MappedFile> mapped(new MappedFile());
    mapped->base = (uchar*)p;
    mapped->length = len;
    return mapped;
#else
    CV_UNUSED(path);
    return Ptr<MappedFile>();
#endif
}

CV__DNN_INLINE_NS_END
}}
