// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// ENGINE_OPENVINO: readNetFromONNX() hands the whole model to the OpenVINO runtime, which reads,
// compiles and runs it. The Net keeps an empty main graph that only lists the model outputs, so
// that empty(), getUnconnectedOutLayersNames() and the dnn::Model classes work as usual.

#include "precomp.hpp"

#include "net_impl.hpp"

#ifdef OPENCV_DNN_OPENVINO

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <map>

#if defined(_WIN32) && defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
#include <windows.h>
#endif

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

struct OpenVINOEngine
{
    // read once; setPreferableTarget() only drops the compiled model, never this one
    std::shared_ptr<const ov::Model> model;
    // the OpenVINO device the model is compiled for: AUTO lets OpenVINO choose it, until
    // setPreferableTarget() fixes one
    std::string device = "AUTO";
    ov::CompiledModel compiledModel;
    ov::InferRequest request;
    bool compiled = false;
    std::string runningOn;  // the execution devices last logged, to log when AUTO moves

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

// The OpenVINO device for a target set with setPreferableTarget(), or false if ENGINE_OPENVINO
// does not map the target. No precision is passed to OpenVINO, so each device runs at its
// default inference precision: f16 on the GPU (for both GPU targets), f32 on most CPUs, bf16 on
// CPUs with AMX.
static bool openVINODevice(int target, std::string& device)
{
    switch (target)
    {
    case DNN_TARGET_CPU:         device = "CPU"; return true;
    case DNN_TARGET_OPENCL:
    case DNN_TARGET_OPENCL_FP16: device = "GPU"; return true;
    case DNN_TARGET_NPU:         device = "NPU"; return true;
    default:                     return false;
    }
}

// openvino::checkTarget() accepts any GPU that OpenVINO lists, and the GPU plugin lists
// non-Intel GPUs too, through OpenCL. "GPU" without an index is the plugin's first device,
// which is an Intel GPU whenever the machine has one, so that is the device to check.
static bool isOpenVINOTargetAvailable(int target)
{
    try
    {
        if (!openvino::checkTarget((Target)target))
            return false;
        if (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16)
        {
            const std::string arch = getCore("GPU").get_property("GPU", ov::device::architecture);
            if (arch.find("vendor=0x8086") == std::string::npos)
            {
                CV_LOG_INFO(NULL, "DNN/OpenVINO: the first OpenVINO GPU is not an Intel GPU (" << arch << ")");
                return false;
            }
        }
        return true;
    }
    catch (const std::exception& e)
    {
        CV_LOG_INFO(NULL, "DNN/OpenVINO: checking target " << target << " failed: " << e.what());
        return false;
    }
}

void Net::Impl::setOpenVINOTarget(int targetId)
{
    CV_Assert(ov_engine);
    std::string device;
    if (!openVINODevice(targetId, device))
    {
        CV_LOG_WARNING(NULL, "DNN/OpenVINO: ENGINE_OPENVINO does not support target " << targetId
                             << ", using DNN_TARGET_CPU instead");
        targetId = DNN_TARGET_CPU;
        device = "CPU";
    }
    else if (targetId != DNN_TARGET_CPU && !isOpenVINOTargetAvailable(targetId))
    {
        CV_LOG_WARNING(NULL, "DNN/OpenVINO: no OpenVINO " << device << " device is available for target "
                             << targetId << ", using DNN_TARGET_CPU instead");
        targetId = DNN_TARGET_CPU;
        device = "CPU";
    }
    preferableTarget = targetId;
    OpenVINOEngine& eng = *ov_engine;
    // DNN_TARGET_OPENCL and DNN_TARGET_OPENCL_FP16 run the same way, so the compiled model is kept
    if (eng.device == device)
        return;
    eng.device = device;

    // The model stays read. Only the compiled model is dropped, so the next forward() compiles
    // it again for the new device without parsing the ONNX file again.
    eng.request = ov::InferRequest();
    eng.compiledModel = ov::CompiledModel();
    eng.compiled = false;
}

// The devices a compiled model runs on, for the log. Right after an AUTO compile this is
// "(CPU)" while the GPU still compiles; AUTO moves to the GPU once it is ready.
static std::string executionDevices(const ov::CompiledModel& compiled)
{
    std::string list;
    try
    {
        for (const std::string& name : compiled.get_property(ov::execution_devices))
            list += (list.empty() ? "" : ",") + name;
    }
    catch (const std::exception&)
    {
    }
    return list.empty() ? std::string("unknown") : list;
}

// Settings read from the environment each time the model is compiled. The cache directory applies
// to every device; the AUTO settings only when no target is set, each to the AUTO property of the
// same purpose (see dnn_engine.markdown).
static const char* const ENV_CACHE_DIR = "OPENCV_DNN_OPENVINO_CACHE_DIR";
static const char* const ENV_AUTO_DEVICES = "OPENCV_DNN_OPENVINO_AUTO_DEVICES";
static const char* const ENV_AUTO_UTILIZATION = "OPENCV_DNN_OPENVINO_AUTO_UTILIZATION_THRESHOLD";
static const char* const ENV_AUTO_PERF_CURVES = "OPENCV_DNN_OPENVINO_AUTO_PERF_CURVES";
static const char* const ENV_AUTO_LOW_POWER_DEVICE = "OPENCV_DNN_OPENVINO_AUTO_LOW_POWER_DEVICE";
static const char* const ENV_AUTO_COMPILE_FOR_ALL = "OPENCV_DNN_OPENVINO_AUTO_COMPILE_FOR_ALL";

static std::string withoutSpaces(std::string text)
{
    text.erase(std::remove_if(text.begin(), text.end(), [](unsigned char c) { return std::isspace(c) != 0; }),
               text.end());
    return text;
}

// "GPU:80,CPU:90": AUTO leaves a device out while its utilization is above its percent
static std::map<std::string, unsigned> parseUtilizationThresholds(const std::string& value)
{
    std::map<std::string, unsigned> thresholds;
    std::istringstream items(withoutSpaces(value));
    std::string item;
    while (std::getline(items, item, ','))
    {
        const size_t colon = item.find(':');
        const std::string percent = colon == std::string::npos ? std::string() : item.substr(colon + 1);
        const bool digits = !percent.empty() && percent.size() <= 3 &&
            std::all_of(percent.begin(), percent.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
        if (colon == 0 || !digits || std::atoi(percent.c_str()) > 100)
            CV_Error_(Error::StsBadArg, ("DNN/OpenVINO: %s: '%s' is not DEVICE:PERCENT with a percent from 0 to 100",
                                         ENV_AUTO_UTILIZATION, item.c_str()));
        thresholds[item.substr(0, colon)] = (unsigned)std::atoi(percent.c_str());
    }
    return thresholds;
}

// A JSON file, read with cv::FileStorage, with one curve per device key ("CPU", "iGPU", "dGPU",
// "NPU"). A curve lists [utilization percent, score] points, and AUTO picks the device with the
// lowest score at its current utilization:
//   { "CPU": [[0, 9.4], [100, 30.0]], "iGPU": [[0, 13.9], [100, 60.0]] }
static std::map<std::string, std::map<unsigned, float> > readPerfCurves(const std::string& path)
{
    FileStorage fs;
    try
    {
        fs.open(path, FileStorage::READ);
    }
    catch (const cv::Exception& e)
    {
        CV_Error_(Error::StsBadArg, ("DNN/OpenVINO: %s: can't parse '%s': %s",
                                     ENV_AUTO_PERF_CURVES, path.c_str(), e.what()));
    }
    if (!fs.isOpened())
        CV_Error_(Error::StsBadArg, ("DNN/OpenVINO: %s: can't open '%s'", ENV_AUTO_PERF_CURVES, path.c_str()));

    std::map<std::string, std::map<unsigned, float> > table;
    const FileNode root = fs.root();
    if (!root.isMap() || root.empty())
        CV_Error_(Error::StsBadArg, ("DNN/OpenVINO: %s: '%s' must map device keys to curves",
                                     ENV_AUTO_PERF_CURVES, path.c_str()));
    for (FileNodeIterator it = root.begin(); it != root.end(); ++it)
    {
        const FileNode curve = *it;
        std::map<unsigned, float>& points = table[curve.name()];
        bool valid = curve.isSeq() && !curve.empty();
        for (FileNodeIterator p = curve.begin(); valid && p != curve.end(); ++p)
        {
            const FileNode point = *p;
            valid = point.isSeq() && point.size() == 2 && (point[0].isInt() || point[0].isReal()) &&
                    (point[1].isInt() || point[1].isReal());
            const double utilization = valid ? (double)point[0] : -1.0;
            valid = valid && utilization >= 0 && utilization <= 100 && utilization == std::floor(utilization);
            if (valid)
                points[(unsigned)utilization] = (float)(double)point[1];
        }
        if (!valid)
            CV_Error_(Error::StsBadArg, ("DNN/OpenVINO: %s: the curve of '%s' must list [utilization, score] "
                                         "points, with whole percents from 0 to 100",
                                         ENV_AUTO_PERF_CURVES, curve.name().c_str()));
    }
    return table;
}

// The properties the model is compiled with. AUTO's own properties go by name, so that this
// builds with OpenVINO versions that do not declare them, and the ones that the AUTO device in
// use does not support are left out with a warning.
static ov::AnyMap compileProperties(ov::Core& core, const std::string& device)
{
    ov::AnyMap properties;
    const std::string cacheDir = utils::getConfigurationParameterString(ENV_CACHE_DIR, "");
    if (!cacheDir.empty())
        properties.emplace(ov::cache_dir(cacheDir));
    if (device != "AUTO")
        return properties;

    // the candidates in order of priority; naming them turns off AUTO's default exclusions
    // (the NPU and non-Intel GPUs)
    const std::string devices = withoutSpaces(utils::getConfigurationParameterString(ENV_AUTO_DEVICES, ""));
    if (!devices.empty())
        properties.emplace(ov::device::priorities(devices));

    struct AutoSetting { const char* env; const char* property; ov::Any value; };
    std::vector<AutoSetting> settings;
    const std::string thresholds = utils::getConfigurationParameterString(ENV_AUTO_UTILIZATION, "");
    if (!thresholds.empty())
        settings.push_back({ENV_AUTO_UTILIZATION, "DEVICES_UTILIZATION_THRESHOLD",
                            parseUtilizationThresholds(thresholds)});
    const std::string curves = utils::getConfigurationParameterString(ENV_AUTO_PERF_CURVES, "");
    if (!curves.empty())
        settings.push_back({ENV_AUTO_PERF_CURVES, "PERF_CURVE_TABLE", readPerfCurves(curves)});
    const std::string lowPower = withoutSpaces(utils::getConfigurationParameterString(ENV_AUTO_LOW_POWER_DEVICE, ""));
    if (!lowPower.empty())
        settings.push_back({ENV_AUTO_LOW_POWER_DEVICE, "LOW_POWER_DEVICE", lowPower});
    if (utils::getConfigurationParameterBool(ENV_AUTO_COMPILE_FOR_ALL, false))
    {
        // AUTO stores the models compiled for the other candidates there
        if (cacheDir.empty())
        {
            CV_LOG_WARNING(NULL, "DNN/OpenVINO: " << ENV_AUTO_COMPILE_FOR_ALL << " needs " << ENV_CACHE_DIR
                                 << ", so it is ignored");
        }
        else
        {
            settings.push_back({ENV_AUTO_COMPILE_FOR_ALL, "COMPILE_FOR_ALL", true});
        }
    }
    // with OPENCV_LOG_LEVEL=DEBUG, AUTO prints its own log too, which shows every device switch
    if (cv::utils::logging::getLogLevel() >= cv::utils::logging::LOG_LEVEL_DEBUG)
        settings.push_back({"OPENCV_LOG_LEVEL", ov::log::level.name(), ov::log::Level::INFO});
    if (settings.empty())
        return properties;

    std::vector<ov::PropertyName> supported;
    try
    {
        supported = core.get_property("AUTO", ov::supported_properties);
    }
    catch (const std::exception& e)
    {
        CV_LOG_WARNING(NULL, "DNN/OpenVINO: can't list the properties of AUTO: " << e.what());
    }
    for (const AutoSetting& s : settings)
    {
        const bool isSupported = std::any_of(supported.begin(), supported.end(),
                                             [&](const std::string& name) { return name == s.property; });
        if (isSupported)
            properties.emplace(s.property, s.value);
        else
            CV_LOG_WARNING(NULL, "DNN/OpenVINO: AUTO in this OpenVINO version does not support " << s.property
                                 << ", so " << s.env << " is ignored");
    }
    return properties;
}

void Net::Impl::finalizeOpenVINO()
{
    CV_Assert(ov_engine);
    OpenVINOEngine& eng = *ov_engine;
    if (eng.compiled)
        return;

    // AUTO unless setPreferableTarget() fixed a device. No precision hint either: the device
    // runs at OpenVINO's default inference precision.
    const std::string device = eng.device;
    ov::Core& core = getCore(device);
    const ov::AnyMap properties = compileProperties(core, device);  // a bad setting throws here
    if (!properties.empty())
    {
        std::string names;
        for (const auto& p : properties)
            names += (names.empty() ? "" : ", ") + p.first;
        CV_LOG_INFO(NULL, "DNN/OpenVINO: compiling for " << device << " with " << names);
    }
    try
    {
        eng.compiledModel = core.compile_model(eng.model, device, properties);
        eng.request = eng.compiledModel.create_infer_request();
    }
    catch (const std::exception& e)
    {
        CV_Error_(Error::StsError, ("DNN/OpenVINO: failed to compile the model for %s%s: %s",
                                    device.c_str(), device == "CPU" ? "" : ", DNN_TARGET_CPU may still work",
                                    e.what()));
    }
    eng.compiled = true;
    eng.runningOn = executionDevices(eng.compiledModel);
    CV_LOG_INFO(NULL, "DNN/OpenVINO: compiled the model for " << device << ", running on " << eng.runningOn);
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

    // AUTO starts on the CPU while the GPU compiles and moves later: log the move. Only when
    // the INFO level is on, so the device query costs nothing otherwise.
    if (eng.device == "AUTO" && cv::utils::logging::getLogLevel() >= cv::utils::logging::LOG_LEVEL_INFO)
    {
        const std::string devices = executionDevices(eng.compiledModel);
        if (devices != eng.runningOn)
        {
            CV_LOG_INFO(NULL, "DNN/OpenVINO: AUTO now runs on " << devices);
            eng.runningOn = devices;
        }
    }

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

#endif  // OPENCV_DNN_OPENVINO
