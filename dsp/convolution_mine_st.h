#pragma once
#include "fft.h"
#include <algorithm>
#include <cstddef>
#include <vector>

class ConvolutionEngine
{
public:
  void loadIR(const float* irData, const size_t irLength);
  void process(const float* input, float* output, const size_t numSamples);
  void reset();

private:
  size_t numIrSegments_;
  size_t currSegment_;
  bool irLoaded_ = false;

  std::vector<std::vector<float>> irSegmentsFFT_;
  std::vector<std::vector<float>> inputHistoryFFT_;
  std::vector<float> overlapBuffer_;

  static constexpr size_t fftSize_ = 256;
  static constexpr size_t blockSize_ = 128;
  // static constexpr size_t segmentSize_ = fftSize_ - blockSize_;
};

class StereoConvolutionReverb
{
public:
  void loadIR(const float* irData, size_t irLengthPerChannel, int numChannels);
  void process(float* left, float* right, int numSamples);
  void reset();

private:
  ConvolutionEngine leftEngine_;
  ConvolutionEngine rightEngine_;
};
