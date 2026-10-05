// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "net_impl.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;
using std::string;

typedef std::pair<int, int> int_pair;
typedef std::pair<int, Arg> int_arg_pair;

static bool isConcreteShape(const MatShape& shape)
{
    return !shape.empty() && !shape.hasSymbols();
}

struct ConstFolding
{
    Net::Impl* netimpl;
    std::vector<int> usecounts;
    std::vector<MatShape> knownShapes;
    std::vector<int> knownTypes;

    ConstFolding(Net::Impl* netimpl_) : netimpl(netimpl_) {}

    void setKnown(Arg arg, const MatShape& shape, int type)
    {
        knownShapes[arg.idx] = shape;
        knownTypes[arg.idx] = type;
    }

    void process()
    {
        size_t nargs = netimpl->args.size();
        netimpl->__tensors__.resize(nargs);
        netimpl->useCounts(usecounts);
        knownShapes.assign(nargs, MatShape());
        knownTypes.assign(nargs, -1);
        const std::vector<Arg>& graphInputs = netimpl->mainGraph->inputs();
        for (size_t i = 0; i < graphInputs.size(); i++) {
            if (i >= netimpl->pinnedInputShapes.size() || !isConcreteShape(netimpl->pinnedInputShapes[i]))
                continue;
            const MatShape& pinned = netimpl->pinnedInputShapes[i];
            Arg inp = graphInputs[i];
            const ArgData& adata = netimpl->args.at(inp.idx);
            if (adata.type < 0)
                continue;
            const Mat& t = netimpl->argTensor(inp);
            if (!t.empty() && t.shape() != pinned)
                continue;
            setKnown(inp, pinned, t.empty() ? adata.type : t.type());
        }
        processGraph(netimpl->mainGraph);
        netimpl->scratchBufs.clear();
    }

    void inferOutputs(const Ptr<LayerInfo>& layer, const std::vector<MatShape>& inpShapes,
                      const std::vector<int>& inpTypes)
    {
        const std::vector<Arg>& outputs = layer->outputs;
        std::vector<MatShape> outShapes, tempShapes;
        std::vector<int> outTypes, tempTypes;
        try {
            layer->getMemoryShapes(inpShapes, (int)outputs.size(), outShapes, tempShapes);
            layer->getTypes(inpTypes, (int)outputs.size(), (int)tempShapes.size(), outTypes, tempTypes);
        } catch (const cv::Exception& e) {
            CV_UNUSED(e);
            CV_LOG_DEBUG(NULL, "DNN/ConstFold: shape inference failed for layer '" << layer->name
                         << "' (" << layer->type << "): " << e.msg);
            return;
        }
        if (outShapes.size() != outputs.size() || outTypes.size() != outputs.size())
            return;
        for (size_t k = 0; k < outputs.size(); k++) {
            if (outputs[k].idx > 0 && outTypes[k] >= 0 && !outShapes[k].hasSymbols())
                setKnown(outputs[k], outShapes[k], outTypes[k]);
        }
    }

    LayerInfo* getLayer(std::vector<Ptr<LayerInfo> >& newprog, int op_idx) const
    {
        return op_idx >= 0 ? newprog.at(op_idx).get() : 0;
    }

    void unuse(Arg inp)
    {
        CV_Assert(usecounts[inp.idx] > 0);
        if (--usecounts[inp.idx] == 0 && netimpl->isConstArg(inp)) {
            netimpl->__tensors__[inp.idx] = Mat(); // deallocate unused tensor
        }
    }

    // Folding materialises an op's result as one constant tensor and drops its inputs.
    // That is a win when the result is no bigger than what it replaces, and a loss when
    // it is much bigger -- the LoRA case, where base + (A @ B) * scale folds a pair of
    // low-rank factors into a full-rank weight and throws away the reason LoRA is small.
    //
    // Skipped only when the result is both large in absolute terms and much larger than
    // its inputs, so ordinary graphs (whose folds are small) are untouched. Tunable with
    // OPENCV_DNN_CONSTFOLD_MIN_BYTES and OPENCV_DNN_CONSTFOLD_MAX_GROWTH; growth 0
    // restores unconditional folding.
    static size_t foldMinBytes()
    {
        static const size_t v = (size_t)utils::getConfigurationParameterSizeT(
            "OPENCV_DNN_CONSTFOLD_MIN_BYTES", (size_t)16 << 20);
        return v;
    }

    static double foldMaxGrowth()
    {
        static const double v = utils::getConfigurationParameterSizeT(
            "OPENCV_DNN_CONSTFOLD_MAX_GROWTH", 4);
        return v;
    }

    static size_t shapeBytes(const MatShape& shape, int type)
    {
        return type < 0 ? 0 : shape.total() * CV_ELEM_SIZE(type);
    }

    bool foldInflatesMemory(const Ptr<LayerInfo>& layer,
                            const std::vector<MatShape>& inpShapes,
                            const std::vector<int>& inpTypes,
                            size_t noutputs) const
    {
        const double growth = foldMaxGrowth();
        if (growth <= 0)
            return false;

        std::vector<MatShape> outShapes, tempShapes;
        std::vector<int> outTypes, tempTypes;
        try {
            layer->getMemoryShapes(inpShapes, (int)noutputs, outShapes, tempShapes);
            layer->getTypes(inpTypes, (int)noutputs, (int)tempShapes.size(), outTypes, tempTypes);
        } catch (const cv::Exception& e) {
            CV_UNUSED(e);
            return false;   // can't size it -> fold as before
        }
        if (outShapes.size() != noutputs || outTypes.size() != noutputs)
            return false;

        size_t outBytes = 0, inpBytes = 0;
        for (size_t k = 0; k < noutputs; k++) {
            if (outShapes[k].hasSymbols())
                return false;
            outBytes += shapeBytes(outShapes[k], outTypes[k]);
        }
        for (size_t k = 0; k < inpShapes.size(); k++)
            inpBytes += shapeBytes(inpShapes[k], inpTypes[k]);

        if (outBytes < foldMinBytes())
            return false;
        if ((double)outBytes <= (double)inpBytes * growth)
            return false;

        CV_LOG_INFO(NULL, cv::format(
            "DNN/ConstFold: skipping fold of '%s' (%s): %.1f MiB out vs %.1f MiB in (%.1fx)",
            layer->name.c_str(), layer->type.c_str(),
            outBytes / 1048576.0, inpBytes / 1048576.0,
            inpBytes ? (double)outBytes / (double)inpBytes : 0.0));
        return true;
    }

    bool processGraph(Ptr<Graph>& graph)
    {
        netimpl->scratchBufs.clear();
        bool modified = false;
        const std::vector<Ptr<LayerInfo> >& prog = graph->prog();
        size_t i, nops = prog.size();
        std::vector<Ptr<LayerInfo> > newprog;
        std::vector<Arg> removed_args;
        std::vector<Mat> inpMats, tempMats;
        std::vector<int> inpTypes, outTypes, tempTypes;
        std::vector<MatShape> inpShapes, outShapes, tempShapes;

        for (i = 0; i < nops; i++) {
            const Ptr<LayerInfo>& layer = prog[i];
            std::vector<Ptr<Graph> >* subgraphs = layer->subgraphs();
            if (subgraphs) {
                for (Ptr<Graph>& g: *subgraphs) {
                    if (processGraph(g))
                        modified = true;
                }
                newprog.push_back(layer);
                continue;
            }
            const std::vector<Arg>& inputs = layer->inputs;
            const std::vector<Arg>& outputs = layer->outputs;
            size_t j, ninputs = inputs.size(), noutputs = outputs.size();
            bool all_const = true, all_known = true;
            inpMats.assign(ninputs, Mat());
            inpTypes.resize(ninputs);
            inpShapes.resize(ninputs);
            for (j = 0; j < ninputs; j++) {
                Arg inp = inputs[j];
                bool const_arg = netimpl->isConstArg(inp);
                if (!const_arg)
                    all_const = false;
                if (all_const) {
                    const Mat& m = netimpl->argTensor(inp);
                    inpMats[j] = m;
                    inpTypes[j] = m.type();
                    inpShapes[j] = m.shape();
                } else if (const_arg) {
                    const Mat& m = netimpl->argTensor(inp);
                    inpTypes[j] = m.type();
                    inpShapes[j] = m.shape();
                } else if (knownTypes[inp.idx] >= 0) {
                    inpTypes[j] = knownTypes[inp.idx];
                    inpShapes[j] = knownShapes[inp.idx];
                } else {
                    all_known = false;
                }
            }
            const bool fold_shape = !all_const && all_known && ninputs == 1 && layer->type == "Shape";
            if (fold_shape)
                inpMats[0] = Mat(inpShapes[0], inpTypes[0], (void*)nullptr);

            // A Shape fold is a handful of ints; only real tensor folds can inflate.
            const bool skip_fold = all_const && !fold_shape &&
                                   foldInflatesMemory(layer, inpShapes, inpTypes, noutputs);

            if ((all_const || fold_shape) && !skip_fold /*&&
                op->supportBlockLayout(0, (int)ninputs) <= 0 // we don't currently support constant folding
                                               // for block-layout operations (Convolution, MaxPool, AveragePool)
                */) {
                // Use a fresh vector of Mat's for outputs since we want to make these outputs the new constant tensors.
                // So, they must be unique and don't interfere with other tensors.
                std::vector<Mat> outMats(noutputs);
                std::vector<std::pair<uchar*, size_t> > outOrigData;
                if (!layer->dynamicOutputShapes())
                    netimpl->allocateLayerOutputs(layer, inpTypes, inpShapes, outTypes,
                                                  outShapes, outOrigData, outMats, tempTypes, tempShapes, tempMats,
                                                  netimpl->scratchBufs, false);
                Ptr<Layer> execLayer = layer.dynamicCast<Layer>();
                CV_Assert(execLayer);  // const-folded ops are CPU-executable (monolithic) layers
                execLayer->finalize(inpMats, outMats);
                execLayer->forward(inpMats, outMats, tempMats);
                CV_Assert(outMats.size() == noutputs);
                for (j = 0; j < noutputs; j++) {
                    Arg out = outputs[j];
                    ArgData& out_data = netimpl->args.at(out.idx);
                    const Mat& m = outMats[j];
                    out_data.type = m.type();
                    out_data.shape = m.shape();
                    out_data.kind = DNN_ARG_CONST; // re-classify each output as constant
                    netimpl->__tensors__.at(out.idx) = m;
                    if (out.idx > 0)
                        setKnown(out, m.shape(), m.type());
                }

                modified = true;
                for (size_t i = 0; i < ninputs; i++)
                    unuse(inputs[i]);
                //printf("folded %s: %s\n", op->name().data(), node->name().data());
                // we don't add operation into the new program,
                // because the output of the all-const inputs operation is now a constant,
                // stored in a separate tensor
            } else {
                if (all_known && !layer->dynamicOutputShapes())
                    inferOutputs(layer, inpShapes, inpTypes);
                newprog.push_back(layer);
            }
        }

        if (modified) {
            graph->setProg(newprog);
        }

        return modified;
    }
};

void Net::Impl::constFold()
{
    ConstFolding constfolder(this);
    constfolder.process();
}

CV__DNN_INLINE_NS_END
}}
