// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// ENGINE_OPENVINO on each DNN target, in three steps:
//   device_discovery: a new ov::Core lists its devices (plugins found, loaded and queried)
//   first_forward:    a new Net reads the model, compiles it for the device and runs it once
//   forward:          setInput() and forward() on a Net whose model is already compiled
// Every test records the OpenVINO version it ran with, so results of builds against different
// OpenVINO packages can be told apart and compared with modules/ts/misc/summary.py.

#include "perf_precomp.hpp"

#ifdef HAVE_INF_ENGINE

#include <openvino/openvino.hpp>

namespace opencv_test {

// The headers used at build time and the openvino.dll loaded at run time must come from the
// same OpenVINO package; another openvino.dll found first on PATH would give numbers of
// another version. Returns an error message, empty when the versions match.
static std::string checkOpenVINOVersion()
{
    const std::string built = cv::format("%d.%d.%d", OPENVINO_VERSION_MAJOR, OPENVINO_VERSION_MINOR,
                                         OPENVINO_VERSION_PATCH);
    const std::string loaded = ov::get_openvino_version().buildNumber;
    ::testing::Test::RecordProperty("openvino_built", built);
    ::testing::Test::RecordProperty("openvino_loaded", loaded);
    if (loaded.compare(0, built.size(), built) != 0)
        return "built against OpenVINO " + built + " but loaded " + loaded + ", check PATH";
    return std::string();
}

// target name in the test name -> DNN target
static int targetByName(const std::string& name)
{
    if (name == "GPU_FP32")
        return DNN_TARGET_OPENCL;
    if (name == "GPU_FP16")
        return DNN_TARGET_OPENCL_FP16;
    if (name == "NPU")
        return DNN_TARGET_NPU;
    return DNN_TARGET_CPU;
}

// Whether OpenVINO has a device for the target, checked as ENGINE_OPENVINO checks it, so a
// target that would fall back to the CPU is skipped instead of measured as the CPU.
static bool haveOpenVINODevice(int target)
{
    const std::string device = target == DNN_TARGET_NPU ? "NPU"
                             : (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16) ? "GPU"
                             : "CPU";
    try
    {
        ov::Core core;
        bool found = false;
        for (const std::string& name : core.get_available_devices())
            found = found || name.compare(0, device.size(), device) == 0;
        if (found && device == "GPU")
            found = core.get_property("GPU", ov::device::architecture).find("vendor=0x8086") != std::string::npos;
        return found;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

// input shape of each benchmark model, from opencv_extra/testdata/dnn/onnx/models
static std::vector<int> modelInputShape(const std::string& model)
{
    if (model == "xfeat")
        return {1, 1, 640, 640};  // grayscale image
    if (model == "yolov10s")
        return {1, 3, 480, 640};
    return {1, 3, 640, 640};
}

static Net readOpenVINONet(const std::string& model)
{
    Net net = readNetFromONNX(findDataFile("dnn/onnx/models/" + model + ".onnx", false), ENGINE_OPENVINO);
    // readNetFromONNX() falls back to ENGINE_OPENCV when ENGINE_OPENVINO is not available, and
    // OPENCV_FORCE_DNN_ENGINE overrides the choice; ENGINE_OPENVINO leaves the graph empty
    if (!net.getMainGraph()->prog().empty())
        throw SkipTestException("ENGINE_OPENVINO did not read the model");
    return net;
}

PERF_TEST(DNN_OpenVINO_Engine, device_discovery)
{
    const std::string versionError = checkOpenVINOVersion();
    ASSERT_TRUE(versionError.empty()) << versionError;

    std::vector<std::string> devices;
    TEST_CYCLE_N(10)
    {
        ov::Core core;
        devices = core.get_available_devices();
    }

    std::string list;
    for (const std::string& name : devices)
        list += (list.empty() ? "" : ",") + name;
    RecordProperty("openvino_devices", list);
    SANITY_CHECK_NOTHING();
}

// model file name without ".onnx", target name
typedef TestBaseWithParam<tuple<std::string, std::string> > DNN_OpenVINO_Target;

PERF_TEST_P_(DNN_OpenVINO_Target, first_forward)
{
    const std::string versionError = checkOpenVINOVersion();
    ASSERT_TRUE(versionError.empty()) << versionError;

    const std::string model = get<0>(GetParam());
    const int target = targetByName(get<1>(GetParam()));
    if (!haveOpenVINODevice(target))
        throw SkipTestException("OpenVINO has no device for this target");
    readOpenVINONet(model);  // skips the test before timing if the engine is not available

    Mat input(modelInputShape(model), CV_32F);
    randu(input, 0.f, 1.f);

    TEST_CYCLE_N(5)
    {
        Net net = readNetFromONNX(findDataFile("dnn/onnx/models/" + model + ".onnx", false), ENGINE_OPENVINO);
        net.setPreferableTarget(target);
        net.setInput(input);
        Mat out = net.forward();
    }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(DNN_OpenVINO_Target, forward)
{
    const std::string versionError = checkOpenVINOVersion();
    ASSERT_TRUE(versionError.empty()) << versionError;

    const std::string model = get<0>(GetParam());
    const int target = targetByName(get<1>(GetParam()));
    if (!haveOpenVINODevice(target))
        throw SkipTestException("OpenVINO has no device for this target");
    Net net = readOpenVINONet(model);
    net.setPreferableTarget(target);

    Mat input(modelInputShape(model), CV_32F);
    randu(input, 0.f, 1.f);
    net.setInput(input);
    net.forward();  // compiles the model for the device

    TEST_CYCLE()
    {
        net.setInput(input);
        Mat out = net.forward();
    }
    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, DNN_OpenVINO_Target, testing::Combine(
    testing::Values(std::string("yolo26n"), std::string("yolov10s"), std::string("xfeat")),
    testing::Values(std::string("CPU"), std::string("GPU_FP32"), std::string("GPU_FP16"), std::string("NPU"))
));

} // namespace opencv_test

#endif  // HAVE_INF_ENGINE
