#include "../convolution_mine_st.h"
#include "../convolution_mt.h"
#include "../fft.h"
#include <cmath>
#include <cstdio>
#include <vector>

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

static int sTestsPassed = 0;
static int sTestsFailed = 0;

static void check(bool condition, const char* label)
{
  if (condition) {
    printf("  PASS: %s\n", label);
    ++sTestsPassed;
  } else {
    printf("  FAIL: %s\n", label);
    ++sTestsFailed;
  }
}

static float maxDiff(const float* a, const float* b, size_t len)
{
  float max = 0.0f;
  for (size_t i = 0; i < len; ++i)
    max = std::fmax(max, std::fabs(a[i] - b[i]));
  return max;
}

// Build level-0 IR FFTs — Gardner uses 2 partitions covering IR[0..2B)
static std::vector<std::vector<float>> buildLevel0IrFFT(const std::vector<float>& ir,
                                                         size_t blockSize)
{
  size_t fftSize = blockSize * 2;
  std::vector<std::vector<float>> partitions;
  for (size_t p = 0; p < 2; ++p) {
    std::vector<float> slice(fftSize, 0.0f);
    size_t offset = p * blockSize;
    size_t count = (offset < ir.size())
                     ? std::min(blockSize, ir.size() - offset) : 0;
    for (size_t i = 0; i < count; ++i)
      slice[i] = ir[offset + i];
    std::vector<float> irFFT(fftSize * 2, 0.0f);
    fft(slice.data(), irFFT.data(), fftSize);
    partitions.push_back(std::move(irFFT));
  }
  return partitions;
}

// Run level0 + TailEngine for numBlocks and return the combined output
static std::vector<float> runMT(const std::vector<float>& input,
                                 const std::vector<float>& ir,
                                 size_t blockSize,
                                 size_t numBlocks)
{
  auto level0IrFFT = buildLevel0IrFFT(ir, blockSize);

  ConvolutionLevel level0(0, blockSize, 0, 2);
  TailEngine tail;
  tail.loadIR(ir.data(), ir.size(), blockSize);

  std::vector<float> output(blockSize * numBlocks, 0.0f);

  for (size_t block = 0; block < numBlocks; ++block) {
    const float* in  = input.data() + block * blockSize;
    float*       out = output.data() + block * blockSize;

    level0.processBlock(in, level0IrFFT);
    tail.processBlock(in);

    const float* level0Out  = level0.getResult();
    const float* tl  = tail.getResult();
    for (size_t i = 0; i < blockSize; ++i)
      out[i] = level0Out[i] + tl[i];
  }

  return output;
}

// Run single-threaded ConvolutionEngine for numBlocks
static std::vector<float> runST(const std::vector<float>& input,
                                 const std::vector<float>& ir,
                                 size_t blockSize,
                                 size_t numBlocks)
{
  ConvolutionEngine engine;
  engine.loadIR(ir.data(), ir.size());

  std::vector<float> output(blockSize * numBlocks, 0.0f);

  for (size_t block = 0; block < numBlocks; ++block) {
    const float* in  = input.data() + block * blockSize;
    float*       out = output.data() + block * blockSize;
    engine.process(in, out, blockSize);
  }

  return output;
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

// Impulse input, short IR spanning 1 segment — only level 0 fires, tail is silent
static void testImpulseShortIR()
{
  printf("\n[testImpulseShortIR]\n");

  const size_t blockSize = 128;
  const size_t numBlocks = 4;

  std::vector<float> ir = { 1.0f, 0.5f, 0.25f, 0.125f };

  std::vector<float> input(blockSize * numBlocks, 0.0f);
  input[0] = 1.0f;

  auto mt = runMT(input, ir, blockSize, numBlocks);
  auto st = runST(input, ir, blockSize, numBlocks);

  float diff = maxDiff(mt.data(), st.data(), blockSize * numBlocks);
  printf("  max diff MT vs ST: %e\n", diff);
  check(diff < 1e-3f, "impulse + short IR: MT matches ST");
}

// Impulse input, IR spanning multiple segments — exercises the tail engine
static void testImpulseMultiSegmentIR()
{
  printf("\n[testImpulseMultiSegmentIR]\n");

  const size_t blockSize = 128;
  const size_t numBlocks = 16;
  const size_t irLen = blockSize * 4; // spans levels 0, 1, 2

  std::vector<float> ir(irLen, 0.0f);
  for (size_t i = 0; i < irLen; ++i)
    ir[i] = std::exp(-3.0f * static_cast<float>(i) / static_cast<float>(irLen));

  std::vector<float> input(blockSize * numBlocks, 0.0f);
  input[0] = 1.0f;

  auto mt = runMT(input, ir, blockSize, numBlocks);
  auto st = runST(input, ir, blockSize, numBlocks);

  float diff = maxDiff(mt.data(), st.data(), blockSize * numBlocks);
  printf("  max diff MT vs ST: %e\n", diff);
  check(diff < 1e-3f, "impulse + multi-segment IR: MT matches ST");
}

// Sine wave input, IR spanning multiple segments
static void testSineMultiSegmentIR()
{
  printf("\n[testSineMultiSegmentIR]\n");

  const size_t blockSize = 128;
  const size_t numBlocks = 16;
  const size_t irLen = blockSize * 4;

  std::vector<float> ir(irLen, 0.0f);
  ir[0] = 1.0f;
  ir[blockSize] = 0.5f;
  ir[blockSize * 2] = 0.25f;
  ir[blockSize * 3] = 0.125f;

  std::vector<float> input(blockSize * numBlocks);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = std::sin(2.0f * M_PI * 440.0f * i / 44100.0f);

  auto mt = runMT(input, ir, blockSize, numBlocks);
  auto st = runST(input, ir, blockSize, numBlocks);

  float diff = maxDiff(mt.data(), st.data(), blockSize * numBlocks);
  printf("  max diff MT vs ST: %e\n", diff);
  check(diff < 1e-3f, "sine + multi-segment IR: MT matches ST");
}

// ----------------------------------------------------------------------------

int main()
{
  printf("=== MT vs ST comparison tests ===\n");

  testImpulseShortIR();
  testImpulseMultiSegmentIR();
  testSineMultiSegmentIR();

  printf("\n%d passed, %d failed\n", sTestsPassed, sTestsFailed);
  return sTestsFailed > 0 ? 1 : 0;
}
