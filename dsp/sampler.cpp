#include "convolution_mt.h"
#include <algorithm>
#include <cstddef>
#include <emscripten/bind.h>
#include <vector>

class Sampler
{
public:
  Sampler() = default;

  void loadSample(uintptr_t samplePtr, size_t sampleLength)
  {
    sampleData_ = reinterpret_cast<float*>(samplePtr);
    sampleLength_ = sampleLength;
    samplePosition_ = sampleLength;
  }

  void prepareLevel0(uintptr_t irPtr, size_t irLength, int numChannels)
  {
    const float* irData = reinterpret_cast<const float*>(irPtr);

    std::vector<float> leftIR(irLength);
    std::vector<float> rightIR(irLength);
    if (numChannels == 1) {
      leftIR.assign(irData, irData + irLength);
      rightIR.assign(irData, irData + irLength);
    } else {
      for (size_t i = 0; i < irLength; ++i) {
        leftIR[i] = irData[i * 2];
        rightIR[i] = irData[i * 2 + 1];
      }
    }

    // normalize IR
    normalizeEnergy(leftIR);
    normalizeEnergy(rightIR);

    // Build level 0 IR FFTs — Gardner uses 2 partitions covering IR[0..2B)
    const size_t fftSize = blockSize_ * 2;
    auto buildLevel0FFTs = [&](const std::vector<float>& ir) {
      std::vector<std::vector<float>> partitions;
      for (size_t p = 0; p < 2; ++p) {
        std::vector<float> slice(fftSize, 0.0f);
        size_t offset = p * blockSize_;
        size_t count =
          (offset < ir.size()) ? std::min(blockSize_, ir.size() - offset) : 0;
        for (size_t i = 0; i < count; ++i)
          slice[i] = ir[offset + i];
        std::vector<float> irFFT(fftSize * 2, 0.0f);
        fft(slice.data(), irFFT.data(), fftSize);
        partitions.push_back(std::move(irFFT));
      }
      return partitions;
    };
    irFFTLevel0Left_ = buildLevel0FFTs(leftIR);
    irFFTLevel0Right_ = buildLevel0FFTs(rightIR);
    level0Left_ = ConvolutionLevel(0, blockSize_, 0, 2);
    level0Right_ = ConvolutionLevel(0, blockSize_, 0, 2);

    level0ResultLeft_.assign(blockSize_, 0.0f);
    level0ResultRight_.assign(blockSize_, 0.0f);
    irReady_ = true;
  }

  void prepare(float sampleRate)
  {
    samplePosition_ = 0;
  }

  void trigger() { samplePosition_ = 0; }

  // Returns pointer to saved dry block (mono, since L/R are identical)
  uintptr_t getDryBlock() const
  {
    return reinterpret_cast<uintptr_t>(dryBlock_.data());
  }

  uintptr_t getLevel0Left() const
  {
    return reinterpret_cast<uintptr_t>(level0ResultLeft_.data());
  }

  uintptr_t getLevel0Right() const
  {
    return reinterpret_cast<uintptr_t>(level0ResultRight_.data());
  }

  void process(uintptr_t leftPtr, uintptr_t rightPtr, int numSamples)
  {
    float* left = reinterpret_cast<float*>(leftPtr);
    float* right = reinterpret_cast<float*>(rightPtr);

    // fill dry signal
    for (int i = 0; i < numSamples; ++i) {
      float sample = 0.0f;
      if (samplePosition_ < sampleLength_) {
        sample = sampleData_[samplePosition_];
        ++samplePosition_;
      }
      left[i] = sample;
      right[i] = sample;
    }

    // save dry block for worker to read via getDryBlock()
    std::copy(left, left + numSamples, dryBlock_.begin());

    // Run level 0 convolution on audio thread
    if (irReady_) {
      level0Left_.processBlock(left, irFFTLevel0Left_);
      level0Right_.processBlock(left, irFFTLevel0Right_);
      const float* level0OutL = level0Left_.getResult();
      const float* level0OutR = level0Right_.getResult();
      std::copy(level0OutL, level0OutL + blockSize_, level0ResultLeft_.begin());
      std::copy(level0OutR, level0OutR + blockSize_, level0ResultRight_.begin());
    }
  }

private:
  static constexpr size_t blockSize_ = 128;

  float* sampleData_ = nullptr;
  size_t sampleLength_ = 0;
  size_t samplePosition_ = 0;


  // dry block saved for worker to read via getDryBlock()
  std::vector<float> dryBlock_ = std::vector<float>(blockSize_, 0.0f);

  // Level 0 convolution (runs on audio thread)
  bool irReady_ = false;
  ConvolutionLevel level0Left_{ 0, blockSize_ };
  ConvolutionLevel level0Right_{ 0, blockSize_ };
  std::vector<std::vector<float>> irFFTLevel0Left_;
  std::vector<std::vector<float>> irFFTLevel0Right_;
  std::vector<float> level0ResultLeft_;
  std::vector<float> level0ResultRight_;
};

EMSCRIPTEN_BINDINGS(audio_module)
{
  emscripten::class_<Sampler>("Sampler")
    .constructor()
    .function("loadSample", &Sampler::loadSample)
    .function("prepareLevel0", &Sampler::prepareLevel0)
    .function("trigger", &Sampler::trigger)
    .function("prepare", &Sampler::prepare)
    .function("process", &Sampler::process)
.function("getDryBlock", &Sampler::getDryBlock)
    .function("getLevel0Left", &Sampler::getLevel0Left)
    .function("getLevel0Right", &Sampler::getLevel0Right);
}
