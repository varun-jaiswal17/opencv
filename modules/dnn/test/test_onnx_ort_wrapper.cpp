// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "npy_blob.hpp"

#ifdef HAVE_ONNXRUNTIME
#include <fstream>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#endif
#endif

namespace opencv_test { namespace {

#ifdef HAVE_ONNXRUNTIME

static std::string _tf(const std::string& filename, bool required = true)
{
    return findDataFile(std::string("dnn/onnx/") + filename, required);
}

static cv::dnn::Net readNetFromONNX_ORT(const std::string& onnxModelPath)
{
    cv::dnn::Net net = cv::dnn::readNetFromONNX(onnxModelPath, cv::dnn::ENGINE_ORT);
    EXPECT_FALSE(net.empty());
    return net;
}

TEST(Test_ONNX_ORT_Wrapper, SingleInputSingleOutput)
{
    const std::string basename = "convolution";
    const std::string onnxmodel = _tf("models/" + basename + ".onnx", true);

    cv::Mat input = blobFromNPY(_tf("data/input_" + basename + ".npy"));
    cv::Mat ref = blobFromNPY(_tf("data/output_" + basename + ".npy"));

    cv::dnn::Net net = readNetFromONNX_ORT(onnxmodel);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    net.setInput(input);
    cv::Mat out = net.forward();

    normAssert(ref, out, "ORT 1in/1out convolution", 1e-5, 1e-4);
}

TEST(Test_ONNX_ORT_Wrapper, MultipleInputSingleOutput)
{
    const std::string basename = "min";
    const std::string onnxmodel = _tf("models/" + basename + ".onnx", true);

    cv::Mat inp0 = blobFromNPY(_tf("data/input_" + basename + "_0.npy"));
    cv::Mat inp1 = blobFromNPY(_tf("data/input_" + basename + "_1.npy"));
    cv::Mat ref = blobFromNPY(_tf("data/output_" + basename + ".npy"));

    cv::dnn::Net net = readNetFromONNX_ORT(onnxmodel);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    net.setInput(inp0, "0");
    net.setInput(inp1, "1");
    cv::Mat out = net.forward();

    normAssert(ref, out, "ORT 2in/1out min", 1e-5, 1e-4);
}

TEST(Test_ONNX_ORT_Wrapper, SingleInputMultipleOutput)
{
    const std::string basename = "top_k";
    const std::string onnxmodel = _tf("models/" + basename + ".onnx", true);

    cv::Mat input = cv::dnn::readTensorFromONNX(_tf("data/input_" + basename + ".pb"));
    cv::Mat ref_val = cv::dnn::readTensorFromONNX(_tf("data/output_" + basename + "_0.pb"));
    cv::Mat ref_ind = cv::dnn::readTensorFromONNX(_tf("data/output_" + basename + "_1.pb"));

    cv::dnn::Net net = readNetFromONNX_ORT(onnxmodel);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    net.setInput(input);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, std::vector<std::string>{"values", "indices"});
    ASSERT_EQ(outputs.size(), 2u);

    normAssert(ref_val, outputs[0], "ORT top_k values", 1e-5, 1e-4);
    normAssert(ref_ind, outputs[1], "ORT top_k indices", 0.0, 0.0);
}

// A directory name that exercises all three UTF-8 encoding widths:
//   "café"  -> 2-byte sequence (U+00E9)
//   "тест" -> 2-byte sequences (Cyrillic)
//   "模型"  -> 3-byte sequences (CJK)
// Written as explicit byte escapes, split across adjacent literals so an escape
// can never swallow the character that follows it. This keeps the test correct
// regardless of how the compiler is told to interpret the source encoding.
static const char kNonAsciiDirName[] =
    "caf" "\xC3\xA9" "_" "\xD1\x82\xD0\xB5\xD1\x81\xD1\x82" "_" "\xE6\xA8\xA1\xE5\x9E\x8B";

// The fixture below cannot be built with cv::utils::fs: createDirectory() and
// exists() go through CreateDirectoryA / GetFileAttributesExA, which read their
// argument in the ANSI codepage and so cannot express these names at all. OpenCV
// holds paths as UTF-8 (see cv::tempfile), so decode properly and use the wide
// API here instead.
#ifdef _WIN32
static std::wstring widenUtf8(const std::string& utf8)
{
    const int len = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                        utf8.c_str(), (int)utf8.size(), NULL, 0);
    if (len <= 0)
        return std::wstring();
    std::wstring wide((size_t)len, L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                        utf8.c_str(), (int)utf8.size(), &wide[0], len);
    return wide;
}
#endif

static bool createDirUtf8(const std::string& utf8)
{
#ifdef _WIN32
    const std::wstring wide = widenUtf8(utf8);
    if (wide.empty())
        return false;
    return ::CreateDirectoryW(wide.c_str(), NULL) != 0
           || ::GetLastError() == ERROR_ALREADY_EXISTS;
#else
    return ::mkdir(utf8.c_str(), 0777) == 0 || errno == EEXIST;
#endif
}

static void removeUtf8(const std::string& utf8, bool isDir)
{
#ifdef _WIN32
    const std::wstring wide = widenUtf8(utf8);
    if (wide.empty())
        return;
    if (isDir)
        ::RemoveDirectoryW(wide.c_str());
    else
        ::DeleteFileW(wide.c_str());
#else
    if (isDir)
        ::rmdir(utf8.c_str());
    else
        ::unlink(utf8.c_str());
#endif
}

static bool copyFileUtf8(const std::string& srcUtf8, const std::string& dstUtf8)
{
    std::ifstream src(srcUtf8.c_str(), std::ios::binary);  // source stays ASCII
#ifdef _WIN32
    const std::wstring dstWide = widenUtf8(dstUtf8);
    if (dstWide.empty())
        return false;
    std::ofstream dst(dstWide.c_str(), std::ios::binary);
#else
    std::ofstream dst(dstUtf8.c_str(), std::ios::binary);
#endif
    if (!src.is_open() || !dst.is_open())
        return false;
    dst << src.rdbuf();
    return dst.good();
}

// Removes the fixture however the test leaves the scope -- an ASSERT_* early
// return, or the exception ORT throws when the path never resolves.
struct SandboxGuard
{
    std::string sandbox, dir, model;
    ~SandboxGuard()
    {
        if (!model.empty())   removeUtf8(model, false);
        if (!dir.empty())     removeUtf8(dir, true);
        if (!sandbox.empty()) removeUtf8(sandbox, true);
    }
};

// Regression test for the ORT path hand-off.
//
// ONNX Runtime takes paths as ORTCHAR_T (wchar_t on Windows). dnn used to widen
// OpenCV's UTF-8 path with std::wstring(s.begin(), s.end()), which sign-extends
// each byte instead of decoding it -- so any model living under a directory with
// non-ASCII characters failed to load with a bare "file doesn't exist". This is
// reachable in the wild simply by having a non-ASCII Windows account name, since
// %TEMP% and most model directories then sit under it.
TEST(Test_ONNX_ORT_Wrapper, LoadFromNonAsciiPath)
{
    const std::string basename = "convolution";
    const std::string srcModel = _tf("models/" + basename + ".onnx", true);

    cv::Mat input = blobFromNPY(_tf("data/input_" + basename + ".npy"));
    cv::Mat ref = blobFromNPY(_tf("data/output_" + basename + ".npy"));

    // Declared before the Net below so that destruction order works in our favour:
    // ORT keeps the model file locked on Windows for as long as the session is
    // alive, and locals are destroyed in reverse, so the session goes first.
    SandboxGuard guard;

    // cv::tempfile() yields an unused, unique, UTF-8 path; use it as our sandbox.
    guard.sandbox = cv::tempfile();
    ASSERT_TRUE(createDirUtf8(guard.sandbox)) << "could not create sandbox " << guard.sandbox;

    guard.dir = guard.sandbox + "/" + kNonAsciiDirName;
    ASSERT_TRUE(createDirUtf8(guard.dir)) << "could not create non-ASCII directory";

    guard.model = guard.dir + "/" + basename + ".onnx";
    ASSERT_TRUE(copyFileUtf8(srcModel, guard.model)) << "could not stage model at " << guard.model;

    cv::dnn::Net net = readNetFromONNX_ORT(guard.model);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // The ORT session is created lazily, so the path only reaches ORT here.
    net.setInput(input);
    cv::Mat out = net.forward();

    normAssert(ref, out, "ORT non-ASCII path convolution", 1e-5, 1e-4);
}

#else  // HAVE_ONNXRUNTIME

TEST(Test_ONNX_ORT_Wrapper, DISABLED_NoONNXRuntime) {}

#endif

}} // namespace
