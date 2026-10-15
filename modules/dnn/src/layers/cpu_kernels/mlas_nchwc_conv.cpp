// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "mlas_nchwc_conv.hpp"

#ifdef HAVE_MLAS
#include "mlas.h"
#include "mlas_gemm.hpp"
#include <opencv2/core/utils/configuration.private.hpp>

namespace cv { namespace dnn {

MlasNchwcMode mlasNchwcMode()
{
    static const MlasNchwcMode mode = []() {
        size_t v = utils::getConfigurationParameterSizeT("OPENCV_DNN_MLAS_NCHWC", (size_t)MLAS_NCHWC_AUTO);
        return v == 0 ? MLAS_NCHWC_OFF : v == 1 ? MLAS_NCHWC_FORCE : MLAS_NCHWC_AUTO;
    }();
    return mode;
}

bool mlasNchwcAvailable(int C0)
{
#if defined(__x86_64__) || defined(_M_X64)
    return mlasAvailable() && C0 > 0 && MlasNchwcGetBlockSize() == (size_t)C0;
#else
    CV_UNUSED(C0);
    return false;
#endif
}

bool mlasNchwcPackFilter(const float* filter, int K, int Cg, int kh, int kw, bool depthwise,
                         std::vector<float>& packed)
{
    const int B = (int)MlasNchwcGetBlockSize();
    if (!filter || K % B != 0 || (!depthwise && Cg % B != 0))
        return false;
    const int64_t shape[4] = {K, Cg, kh, kw};
    packed.resize((size_t)K * Cg * kh * kw);
    if (depthwise)
        MlasReorderFilterOIHWBo(shape, filter, packed.data());
    else
        MlasReorderFilterOIHWBiBo(shape, filter, packed.data());
    return true;
}

bool mlasNchwcConv(const MlasNchwcConvParams& p, const float* input, const float* packedFilter,
                   const float* bias, float* output, bool accumulate)
{
    const int64_t inShape[4] = {p.N, p.C, p.H, p.W};
    const int64_t outShape[4] = {p.N, p.K, p.Ho, p.Wo};
    const int64_t kernel[2] = {p.kh, p.kw};
    const int64_t dilations[2] = {p.dilations[0], p.dilations[1]};
    const int64_t strides[2] = {p.strides[0], p.strides[1]};
    const int64_t pads[4] = {p.pads[0], p.pads[1], p.pads[2], p.pads[3]};

    MLAS_ACTIVATION act;
    act.ActivationKind = MlasIdentityActivation;
    switch (p.activation) {
    case MLAS_NCHWC_ACTIV_NONE:
        break;
    case MLAS_NCHWC_ACTIV_RELU:
        act.ActivationKind = MlasReluActivation;
        break;
    case MLAS_NCHWC_ACTIV_LEAKY_RELU:
        act.ActivationKind = MlasLeakyReluActivation;
        act.Parameters.LeakyRelu.alpha = p.activParams[0];
        break;
    case MLAS_NCHWC_ACTIV_CLIP:
        act.ActivationKind = MlasClipActivation;
        act.Parameters.Clip.minimum = p.activParams[0];
        act.Parameters.Clip.maximum = p.activParams[1];
        break;
    default:
        return false;
    }

    MlasNchwcConv(inShape, kernel, dilations, pads, strides, outShape, (size_t)p.groups,
                  input, packedFilter, bias, output, &act, !accumulate,
                  nullptr, nullptr, false);
    return true;
}

}}

#endif
