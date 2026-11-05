// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Whole-model CPU inference of ONNX networks through each engine of this build.
// ENGINE_OPENCV follows cv::setNumThreads() (--perf_threads); ENGINE_ORT and ENGINE_OPENVINO
// keep their own default number of threads.

#include "perf_precomp.hpp"

namespace opencv_test {

static int engineByName(const std::string& name)
{
    if (name == "ORT")
        return ENGINE_ORT;
    if (name == "OpenVINO")
        return ENGINE_OPENVINO;
    return ENGINE_OPENCV;
}

static std::vector<std::string> perfEngines()
{
    std::vector<std::string> engines(1, "OpenCV");
#ifdef HAVE_ONNXRUNTIME
    engines.push_back("ORT");
#endif
#ifdef HAVE_INF_ENGINE
    engines.push_back("OpenVINO");
#endif
    return engines;
}

// model file (all take a 1x3x224x224 input), engine
typedef TestBaseWithParam<tuple<std::string, std::string> > DNN_ONNX_Engine;

PERF_TEST_P_(DNN_ONNX_Engine, forward)
{
    const std::string model = get<0>(GetParam());
    const int engine = engineByName(get<1>(GetParam()));

    Net net = readNetFromONNX(findDataFile("dnn/onnx/models/" + model, false), engine);
    // readNetFromONNX() falls back to ENGINE_OPENCV when an engine is not available, and
    // OPENCV_FORCE_DNN_ENGINE overrides the choice; the external engines leave the graph empty
    const bool ranByOpenCV = !net.getMainGraph()->prog().empty();
    if (ranByOpenCV != (engine == ENGINE_OPENCV))
        throw SkipTestException("the requested engine is not the one that read the model");

    Mat input({1, 3, 224, 224}, CV_32F);
    randu(input, 0.f, 1.f);
    net.setInput(input);
    net.forward();  // ENGINE_ORT and ENGINE_OPENVINO compile the model here

    TEST_CYCLE()
    {
        Mat res = net.forward();
    }
    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, DNN_ONNX_Engine, testing::Combine(
    testing::Values(std::string("squeezenet.onnx"), std::string("mobilenetv2.onnx"), std::string("resnet50v1.onnx")),
    testing::ValuesIn(perfEngines())
));

} // namespace opencv_test
