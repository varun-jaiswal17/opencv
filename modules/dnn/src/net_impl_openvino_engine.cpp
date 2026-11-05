// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// ENGINE_OPENVINO: readNetFromONNX() hands the whole model to the OpenVINO runtime, which reads,
// compiles and runs it. The Net keeps an empty main graph that only lists the model outputs, so
// that empty(), getUnconnectedOutLayersNames() and the dnn::Model classes work as usual.

#include "precomp.hpp"

#include "net_impl.hpp"

#ifdef HAVE_INF_ENGINE

#include <climits>

#if defined(_WIN32) && defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
#include <windows.h>
#endif

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

struct OpenVINOEngine
{
    std::shared_ptr<const ov::Model> model;
    ov::CompiledModel compiledModel;
    ov::InferRequest request;
    bool compiled = false;

    // per model input, in model order
    std::vector<std::string> inputNames;
    std::vector<ov::element::Type> inputTypes;
    std::vector<Mat> inputs;  // private copies made by setInput(), already in inputTypes

    // every tensor name of every port selects that port
    std::unordered_map<std::string, int> inputIndex;
    std::unordered_map<std::string, int> outputIndex;
    size_t noutputs = 0;
};

#if defined(_WIN32) && defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
static bool widenPath(const std::string& path, std::wstring& wide)
{
    const int len = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                        path.c_str(), (int)path.size(), NULL, 0);
    if (len <= 0)
        return false;
    wide.assign((size_t)len, L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                        path.c_str(), (int)path.size(), &wide[0], len);
    return true;
}
#endif

static std::shared_ptr<ov::Model> readOpenVINOModel(const std::string& path)
{
    ov::Core& core = getCore("CPU");
    try
    {
#if defined(_WIN32) && defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
        // OpenCV paths are UTF-8, so on Windows they go through the wide overload,
        // as toOrtPath() does for ENGINE_ORT
        std::wstring wide;
        if (widenPath(path, wide))
            return core.read_model(wide);
#endif
        return core.read_model(path);
    }
    catch (const std::exception& e)
    {
        CV_Error_(Error::StsError, ("DNN/ONNX/OpenVINO: failed to read '%s': %s", path.c_str(), e.what()));
    }
}

static std::string portName(const ov::Output<const ov::Node>& port, const char* prefix, size_t i)
{
    return port.get_names().empty() ? format("%s_%zu", prefix, i) : port.get_any_name();
}

Net readNetFromONNX2_OpenVINO(const String& onnxFile)
{
    std::shared_ptr<OpenVINOEngine> eng = std::make_shared<OpenVINOEngine>();
    eng->model = readOpenVINOModel(onnxFile);

    const std::vector<ov::Output<const ov::Node> > inputs = eng->model->inputs();
    for (size_t i = 0; i < inputs.size(); i++)
    {
        eng->inputNames.push_back(portName(inputs[i], "input", i));
        eng->inputTypes.push_back(inputs[i].get_element_type());
        for (const std::string& name : inputs[i].get_names())
            eng->inputIndex.emplace(name, (int)i);
    }
    eng->inputs.resize(inputs.size());

    Net net;
    Net::Impl* impl = net.getImpl();
    impl->modelFileName = onnxFile;
    impl->modelFormat = DNN_MODEL_ONNX;

    // List the model outputs on the main graph, so that getUnconnectedOutLayersNames() and the
    // dnn::Model classes see them before the first forward(). The tests check the graph name.
    Ptr<Graph> graph = impl->newGraph("openvino", {}, true);
    const std::vector<ov::Output<const ov::Node> > outputs = eng->model->outputs();
    std::vector<Arg> outArgs;
    for (size_t i = 0; i < outputs.size(); i++)
    {
        for (const std::string& name : outputs[i].get_names())
            eng->outputIndex.emplace(name, (int)i);
        std::string name = portName(outputs[i], "output", i);
        while (impl->haveArg(name))
            name += "_";
        eng->outputIndex.emplace(name, (int)i);
        outArgs.push_back(impl->newArg(name, DNN_ARG_OUTPUT));
    }
    graph->setOutputs(outArgs);
    eng->noutputs = outputs.size();

    impl->ov_engine = eng;
    CV_LOG_INFO(NULL, "DNN/ONNX/OpenVINO: read " << onnxFile << " (" << inputs.size() << " inputs, "
                      << outputs.size() << " outputs)");
    return net;
}

void Net::Impl::finalizeOpenVINO()
{
    CV_Assert(ov_engine);
    OpenVINOEngine& eng = *ov_engine;
    if (eng.compiled)
        return;

    // CPU only for now. Ask for f32 explicitly: the CPU plugin would otherwise run bf16 on CPUs
    // with AMX or AVX512_BF16, and the results would depend on the machine.
    const std::string device = "CPU";
    try
    {
        eng.compiledModel = getCore(device).compile_model(eng.model, device,
                ov::AnyMap{ov::hint::inference_precision(ov::element::f32)});
        eng.request = eng.compiledModel.create_infer_request();
    }
    catch (const std::exception& e)
    {
        CV_Error_(Error::StsError, ("DNN/OpenVINO: failed to compile the model for %s: %s",
                                    device.c_str(), e.what()));
    }
    eng.compiled = true;
}

void Net::Impl::setOpenVINOInput(const Mat& blob, const std::string& name)
{
    CV_Assert(ov_engine);
    OpenVINOEngine& eng = *ov_engine;
    if (blob.empty())
        CV_Error(Error::StsBadArg, "DNN/OpenVINO: input blob is empty");
    CV_CheckEQ(blob.channels(), 1, "DNN/OpenVINO: an input must be a single-channel blob, see blobFromImage()");

    size_t idx = 0;
    if (name.empty())
    {
        if (eng.inputs.size() != 1)
            CV_Error_(Error::StsBadArg, ("DNN/OpenVINO: the model has %zu inputs, so the input name must be given",
                                         eng.inputs.size()));
    }
    else
    {
        auto it = eng.inputIndex.find(name);
        if (it == eng.inputIndex.end())
            CV_Error_(Error::StsObjectNotFound, ("DNN/OpenVINO: input '%s' is not found", name.c_str()));
        idx = (size_t)it->second;
    }

    // forwardOpenVINO() hands this copy to OpenVINO as is, so make it in the model's element type
    const int depth = ovTypeToCvType(eng.inputTypes[idx]);
    if (blob.depth() == depth)
        blob.copyTo(eng.inputs[idx]);
    else
        blob.convertTo(eng.inputs[idx], depth);
}

// As forward() does with ENGINE_OPENCV: a caller's Mat of the right size and type is filled in
// place, otherwise it takes over the private copy.
static void assignOpenVINOOutputs(OutputArrayOfArrays outBlobs, const std::vector<Mat>& outs)
{
    const _InputArray::KindFlag kind = outBlobs.kind();
    if (kind == _InputArray::STD_VECTOR_MAT)
    {
        std::vector<Mat>& dst = outBlobs.getMatVecRef();
        dst.resize(outs.size());
        for (size_t i = 0; i < outs.size(); i++)
        {
            if (!dst[i].empty() && dst[i].size == outs[i].size && dst[i].type() == outs[i].type())
                outs[i].copyTo(dst[i]);
            else
                dst[i] = outs[i];
        }
    }
    else if (kind == _InputArray::STD_VECTOR_UMAT)
    {
        std::vector<UMat>& dst = outBlobs.getUMatVecRef();
        dst.resize(outs.size());
        for (size_t i = 0; i < outs.size(); i++)
            outs[i].copyTo(dst[i]);
    }
    else if (kind == _InputArray::MAT || kind == _InputArray::UMAT)
    {
        CV_CheckEQ(outs.size(), (size_t)1, "DNN/OpenVINO: a single Mat or UMat takes exactly one output");
        outBlobs.assign(outs[0]);
    }
    else
    {
        CV_Error(Error::StsBadArg, "DNN/OpenVINO: outputs must be Mat, UMat, a vector of Mat's or a vector of UMat's");
    }
}

void Net::Impl::forwardOpenVINO(OutputArrayOfArrays outBlobs, const std::vector<std::string>& outNames)
{
    finalizeOpenVINO();
    OpenVINOEngine& eng = *ov_engine;

    std::vector<size_t> outIdxs;
    if (outNames.empty())
    {
        for (size_t i = 0; i < eng.noutputs; i++)
            outIdxs.push_back(i);
    }
    for (const std::string& name : outNames)
    {
        size_t idx = 0;  // forward() without a name returns the first output
        if (!name.empty())
        {
            auto it = eng.outputIndex.find(name);
            if (it == eng.outputIndex.end())
                CV_Error_(Error::StsObjectNotFound, ("DNN/OpenVINO: '%s' is not a model output; "
                                                     "ENGINE_OPENVINO can only return model outputs", name.c_str()));
            idx = (size_t)it->second;
        }
        outIdxs.push_back(idx);
    }

    for (size_t i = 0; i < eng.inputs.size(); i++)
    {
        if (eng.inputs[i].empty())
            CV_Error_(Error::StsError, ("DNN/OpenVINO: input '%s' is not set", eng.inputNames[i].c_str()));
    }

    const int64 start = getTickCount();
    try
    {
        // the tensors wrap the private input copies, OpenVINO reads them in place
        for (size_t i = 0; i < eng.inputs.size(); i++)
        {
            Mat& m = eng.inputs[i];
            eng.request.set_input_tensor(i, ov::Tensor(eng.inputTypes[i], ov::Shape(getShape<size_t>(m)), m.data));
        }
        eng.request.infer();
    }
    catch (const std::exception& e)
    {
        CV_Error_(Error::StsError, ("DNN/OpenVINO: inference failed: %s", e.what()));
    }
    // getPerfProfile() reports the whole model as one entry
    layersTimings.assign(2, 0);
    layersTimings[1] = getTickCount() - start;

    std::vector<Mat> outs;
    outs.reserve(outIdxs.size());
    for (size_t idx : outIdxs)
    {
        ov::Tensor tensor = eng.request.get_output_tensor(idx);
        const ov::Shape shape = tensor.get_shape();  // the actual shape, also for dynamic models
        std::vector<int> dims;
        dims.reserve(shape.size());
        for (size_t d : shape)
        {
            CV_CheckLE(d, (size_t)INT_MAX, "DNN/OpenVINO: output dimension is too large");
            dims.push_back((int)d);
        }
        const void* data = tensor.data();
        // OpenVINO writes the next result into the same buffer, so hand out a copy
        outs.push_back(Mat(dims, ovTypeToCvType(tensor.get_element_type()), const_cast<void*>(data)).clone());
    }
    assignOpenVINOOutputs(outBlobs, outs);
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn

#endif  // HAVE_INF_ENGINE
