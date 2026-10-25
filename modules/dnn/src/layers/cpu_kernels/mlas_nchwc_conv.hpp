// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_MLAS_NCHWC_CONV_HPP
#define OPENCV_DNN_MLAS_NCHWC_CONV_HPP

#include <vector>

namespace cv { namespace dnn {

enum MlasNchwcMode { MLAS_NCHWC_OFF = 0, MLAS_NCHWC_FORCE = 1, MLAS_NCHWC_AUTO = 2 };

enum MlasNchwcActivation {
    MLAS_NCHWC_ACTIV_NONE = 0,
    MLAS_NCHWC_ACTIV_RELU,
    MLAS_NCHWC_ACTIV_LEAKY_RELU,
    MLAS_NCHWC_ACTIV_CLIP
};

struct MlasNchwcConvParams
{
    int N, C, H, W;
    int K, Ho, Wo;
    int kh, kw;
    int strides[2], dilations[2];
    int pads[4];
    int groups;
    int activation;
    float activParams[2];
};

#ifdef HAVE_MLAS

MlasNchwcMode mlasNchwcMode();

bool mlasNchwcAvailable(int C0);

bool mlasNchwcPackFilter(const float* filter, int K, int Cg, int kh, int kw, bool depthwise,
                         std::vector<float>& packed);

bool mlasNchwcConv(const MlasNchwcConvParams& p, const float* input, const float* packedFilter,
                   const float* bias, float* output, bool accumulate);

#else

inline MlasNchwcMode mlasNchwcMode() { return MLAS_NCHWC_OFF; }
inline bool mlasNchwcAvailable(int) { return false; }
inline bool mlasNchwcPackFilter(const float*, int, int, int, int, bool, std::vector<float>&) { return false; }
inline bool mlasNchwcConv(const MlasNchwcConvParams&, const float*, const float*,
                          const float*, float*, bool) { return false; }

#endif

}}

#endif
