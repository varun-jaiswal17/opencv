// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "npy_blob.hpp"

namespace opencv_test { namespace {

#ifdef HAVE_INF_ENGINE

static std::string _tf(const std::string& filename, bool required = true)
{
    return findDataFile(std::string("dnn/onnx/") + filename, required);
}

static int forcedEngine()
{
    return (int)cv::utils::getConfigurationParameterSizeT("OPENCV_FORCE_DNN_ENGINE", cv::dnn::ENGINE_AUTO);
}

static void skipIfOtherEngineForced()
{
    if (forcedEngine() == cv::dnn::ENGINE_OPENCV || forcedEngine() == cv::dnn::ENGINE_ORT)
        throw SkipTestException("OPENCV_FORCE_DNN_ENGINE selects another engine");
}

// readNetFromONNX() falls back to ENGINE_OPENCV, with a warning, when ENGINE_OPENVINO is not
// available (e.g. OpenVINO built as a DNN plugin). Check that the net really runs in OpenVINO,
// so that such a fallback cannot pass these tests.
static cv::dnn::Net readNetFromONNX_OpenVINO(const std::string& onnxModelPath)
{
    skipIfOtherEngineForced();
    cv::dnn::Net net = cv::dnn::readNetFromONNX(onnxModelPath, cv::dnn::ENGINE_OPENVINO);
    EXPECT_FALSE(net.empty());
    Ptr<cv::dnn::Graph> graph = net.getMainGraph();
    EXPECT_TRUE(graph && graph->name() == "openvino") << "the model was not read by ENGINE_OPENVINO";
    return net;
}

TEST(Test_ONNX_OpenVINO_Wrapper, SingleInputSingleOutput)
{
    const std::string basename = "convolution";
    const std::string onnxmodel = _tf("models/" + basename + ".onnx", true);

    cv::Mat input = blobFromNPY(_tf("data/input_" + basename + ".npy"));
    cv::Mat ref = blobFromNPY(_tf("data/output_" + basename + ".npy"));

    cv::dnn::Net net = readNetFromONNX_OpenVINO(onnxmodel);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    net.setInput(input);
    cv::Mat out = net.forward();

    normAssert(ref, out, "OpenVINO 1in/1out convolution", 1e-5, 1e-4);
}

TEST(Test_ONNX_OpenVINO_Wrapper, MultipleInputSingleOutput)
{
    const std::string basename = "min";
    const std::string onnxmodel = _tf("models/" + basename + ".onnx", true);

    cv::Mat inp0 = blobFromNPY(_tf("data/input_" + basename + "_0.npy"));
    cv::Mat inp1 = blobFromNPY(_tf("data/input_" + basename + "_1.npy"));
    cv::Mat ref = blobFromNPY(_tf("data/output_" + basename + ".npy"));

    cv::dnn::Net net = readNetFromONNX_OpenVINO(onnxmodel);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    net.setInput(inp0, "0");
    net.setInput(inp1, "1");
    cv::Mat out = net.forward();

    normAssert(ref, out, "OpenVINO 2in/1out min", 1e-5, 1e-4);
}

TEST(Test_ONNX_OpenVINO_Wrapper, SingleInputMultipleOutput)
{
    const std::string basename = "top_k";
    const std::string onnxmodel = _tf("models/" + basename + ".onnx", true);

    cv::Mat input = cv::dnn::readTensorFromONNX(_tf("data/input_" + basename + ".pb"));
    cv::Mat ref_val = cv::dnn::readTensorFromONNX(_tf("data/output_" + basename + "_0.pb"));
    cv::Mat ref_ind = cv::dnn::readTensorFromONNX(_tf("data/output_" + basename + "_1.pb"));

    cv::dnn::Net net = readNetFromONNX_OpenVINO(onnxmodel);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    net.setInput(input);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, std::vector<std::string>{"values", "indices"});
    ASSERT_EQ(outputs.size(), 2u);

    normAssert(ref_val, outputs[0], "OpenVINO top_k values", 1e-5, 1e-4);
    normAssert(ref_ind, outputs[1], "OpenVINO top_k indices", 0.0, 0.0);
}

// The outputs are known as soon as the model is read: dnn::Model asks for them in its
// constructor. forward() without names returns all of them in model order.
TEST(Test_ONNX_OpenVINO_Wrapper, OutputsKnownBeforeForward)
{
    const std::string basename = "top_k";
    cv::dnn::Net net = readNetFromONNX_OpenVINO(_tf("models/" + basename + ".onnx"));

    std::vector<std::string> names = net.getUnconnectedOutLayersNames();
    ASSERT_EQ(names.size(), 2u);
    EXPECT_EQ(names[0], "values");
    EXPECT_EQ(names[1], "indices");

    cv::Mat input = cv::dnn::readTensorFromONNX(_tf("data/input_" + basename + ".pb"));
    cv::Mat ref_val = cv::dnn::readTensorFromONNX(_tf("data/output_" + basename + "_0.pb"));
    cv::Mat ref_ind = cv::dnn::readTensorFromONNX(_tf("data/output_" + basename + "_1.pb"));
    net.setInput(input);

    std::vector<cv::Mat> all;
    net.forward(all);
    ASSERT_EQ(all.size(), 2u);
    normAssert(ref_val, all[0], "values", 1e-5, 1e-4);
    normAssert(ref_ind, all[1], "indices", 0.0, 0.0);

    std::vector<std::vector<cv::Mat> > perName;
    net.forward(perName, std::vector<std::string>(1, "indices"));
    ASSERT_EQ(perName.size(), 1u);
    ASSERT_EQ(perName[0].size(), 1u);
    normAssert(ref_ind, perName[0][0], "indices by name", 0.0, 0.0);

    // the whole model is reported as one entry
    std::vector<double> timings;
    EXPECT_GT(net.getPerfProfile(timings), 0);
    EXPECT_EQ(timings.size(), 1u);
}

// Only DNN_TARGET_CPU is supported for now; the backend does not matter, OpenVINO runs everything.
TEST(Test_ONNX_OpenVINO_Wrapper, OtherTargetFallsBackToCPU)
{
    const std::string basename = "convolution";
    cv::Mat input = blobFromNPY(_tf("data/input_" + basename + ".npy"));
    cv::Mat ref = blobFromNPY(_tf("data/output_" + basename + ".npy"));

    cv::dnn::Net net = readNetFromONNX_OpenVINO(_tf("models/" + basename + ".onnx"));
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    net.setInput(input);
    cv::Mat out = net.forward();

    normAssert(ref, out, "OpenVINO convolution, CUDA target requested", 1e-5, 1e-4);
}

// #28808 fixed ENGINE_ORT taking over every ONNX net. Nets read with the default engine must
// still run on ENGINE_OPENCV in a build with OpenVINO.
TEST(Test_ONNX_OpenVINO_Wrapper, DefaultEngineUnaffected)
{
    if (forcedEngine() != cv::dnn::ENGINE_AUTO)
        throw SkipTestException("OPENCV_FORCE_DNN_ENGINE is set");

    const std::string basename = "convolution";
    cv::Mat input = blobFromNPY(_tf("data/input_" + basename + ".npy"));
    cv::Mat ref = blobFromNPY(_tf("data/output_" + basename + ".npy"));

    cv::dnn::Net net = cv::dnn::readNetFromONNX(_tf("models/" + basename + ".onnx"));
    Ptr<cv::dnn::Graph> graph = net.getMainGraph();
    ASSERT_TRUE(graph);
    EXPECT_NE(graph->name(), "openvino");
    EXPECT_FALSE(graph->prog().empty());

    net.setInput(input);
    cv::Mat out = net.forward();

    normAssert(ref, out, "ENGINE_AUTO convolution", 1e-5, 1e-4);
}

TEST(Test_ONNX_OpenVINO_Wrapper, Errors)
{
    const std::string basename = "min";
    const std::string onnxmodel = _tf("models/" + basename + ".onnx");
    cv::Mat inp = blobFromNPY(_tf("data/input_" + basename + "_0.npy"));

    cv::dnn::Net net = readNetFromONNX_OpenVINO(onnxmodel);
    EXPECT_THROW(net.forward(), cv::Exception);  // no input set yet
    EXPECT_THROW(net.setInput(inp), cv::Exception);  // two inputs, so a name is needed
    EXPECT_THROW(net.setInput(inp, "no_such_input"), cv::Exception);

    net.setInput(inp, "0");
    net.setInput(inp, "1");
    EXPECT_THROW(net.forward("no_such_output"), cv::Exception);
    EXPECT_NO_THROW(net.forward("2"));

    std::vector<char> buffer;
    readFileContent(onnxmodel, buffer);
    EXPECT_THROW(cv::dnn::readNetFromONNX(buffer.data(), buffer.size(), cv::dnn::ENGINE_OPENVINO), cv::Exception);
    EXPECT_THROW(cv::dnn::readNetFromONNX(onnxmodel + ".missing", cv::dnn::ENGINE_OPENVINO), cv::Exception);
}

#ifndef _WIN32
// #29806 fixed this for ENGINE_ORT. On Windows a UTF-8 path goes through the wide overload of
// read_model(); creating the file there needs a wide path too, so this test is POSIX only.
TEST(Test_ONNX_OpenVINO_Wrapper, NonAsciiPath)
{
    skipIfOtherEngineForced();
    const std::string basename = "convolution";
    cv::Mat input = blobFromNPY(_tf("data/input_" + basename + ".npy"));
    cv::Mat ref = blobFromNPY(_tf("data/output_" + basename + ".npy"));

    std::vector<char> model;
    readFileContent(_tf("models/" + basename + ".onnx"), model);
    const std::string path = cv::tempfile("_mod\xc3\xa8le.onnx");
    {
        std::ofstream f(path.c_str(), std::ios::binary);
        f.write(model.data(), (std::streamsize)model.size());
        ASSERT_TRUE(f.good()) << path;
    }
    cv::dnn::Net net = readNetFromONNX_OpenVINO(path);
    std::remove(path.c_str());

    net.setInput(input);
    cv::Mat out = net.forward();

    normAssert(ref, out, "OpenVINO convolution, non-ASCII path", 1e-5, 1e-4);
}
#endif

// Covers the multi-output path on real detectors: four named detection tensors per graph.
static void runSSDDetectorOpenVINO(const std::string& model)
{
    cv::Mat img = imread(findDataFile("dnn/street.png"));
    ASSERT_FALSE(img.empty());
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, rgb, cv::Size(300, 300));
    if (!rgb.isContinuous()) rgb = rgb.clone();
    int s[] = {1, 300, 300, 3};
    cv::Mat blob(4, s, CV_8U, rgb.data);

    // fetched by download_models.py, so skip when missing
    cv::dnn::Net net = readNetFromONNX_OpenVINO(_tf("models/" + model, false));
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    net.setInput(blob);

    std::vector<cv::Mat> outs;
    net.forward(outs, std::vector<std::string>{
        "detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0"});
    ASSERT_EQ(outs.size(), 4u);
    int nd = static_cast<int>(outs[3].ptr<float>()[0]);
    EXPECT_GE(nd, 1);
}

TEST(Test_ONNX_OpenVINO_Wrapper, SSD_MobileNet_v1)
{
    runSSDDetectorOpenVINO("ssd_mobilenet_v1_coco.onnx");
}

TEST(Test_ONNX_OpenVINO_Wrapper, SSD_MobileNet_v2)
{
    runSSDDetectorOpenVINO("ssd_mobilenet_v2_coco_2018_03_29.onnx");
}

TEST(Test_ONNX_OpenVINO_Wrapper, SSD_Inception_v2)
{
    runSSDDetectorOpenVINO("ssd_inception_v2_coco_2017_11_17.onnx");
}

#else  // HAVE_INF_ENGINE

TEST(Test_ONNX_OpenVINO_Wrapper, DISABLED_NoOpenVINO) {}

#endif

}} // namespace
