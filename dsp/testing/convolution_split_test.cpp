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

// Run split architecture (level0 + TailEngine separately, like real system)
static std::vector<float> runSplit(const std::vector<float>& input,
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

// Run single-threaded ConvolutionEngine
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

// Long IR (many segments, exercises capped partitions)
static void testLongIR_Impulse()
{
  printf("\n[testLongIR_Impulse]\n");

  const size_t blockSize = 128;
  const size_t irLen = blockSize * 100; // 100 segments — well past maxLevel
  const size_t numBlocks = 150;         // enough to see the full tail

  // exponential decay IR
  std::vector<float> ir(irLen, 0.0f);
  for (size_t i = 0; i < irLen; ++i)
    ir[i] = std::exp(-3.0f * static_cast<float>(i) / static_cast<float>(irLen));

  std::vector<float> input(blockSize * numBlocks, 0.0f);
  input[0] = 1.0f;

  auto split = runSplit(input, ir, blockSize, numBlocks);
  auto st    = runST(input, ir, blockSize, numBlocks);

  float diff = maxDiff(split.data(), st.data(), blockSize * numBlocks);
  printf("  max diff split vs ST: %e\n", diff);
  check(diff < 1e-3f, "long IR impulse: split matches ST");
}

// Long IR with continuous sine input
static void testLongIR_Sine()
{
  printf("\n[testLongIR_Sine]\n");

  const size_t blockSize = 128;
  const size_t irLen = blockSize * 100;
  const size_t numBlocks = 150;

  std::vector<float> ir(irLen, 0.0f);
  for (size_t i = 0; i < irLen; ++i)
    ir[i] = std::exp(-3.0f * static_cast<float>(i) / static_cast<float>(irLen));

  std::vector<float> input(blockSize * numBlocks);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = std::sin(2.0f * M_PI * 440.0f * i / 44100.0f);

  auto split = runSplit(input, ir, blockSize, numBlocks);
  auto st    = runST(input, ir, blockSize, numBlocks);

  float diff = maxDiff(split.data(), st.data(), blockSize * numBlocks);
  printf("  max diff split vs ST: %e\n", diff);
  check(diff < 1e-3f, "long IR sine: split matches ST");
}

// Very long IR simulating ~2 seconds at 44100 Hz
static void testVeryLongIR()
{
  printf("\n[testVeryLongIR]\n");

  const size_t blockSize = 128;
  const size_t irLen = 88200; // ~2 seconds
  const size_t numBlocks = 1024;

  std::vector<float> ir(irLen, 0.0f);
  for (size_t i = 0; i < irLen; ++i)
    ir[i] = std::exp(-5.0f * static_cast<float>(i) / static_cast<float>(irLen));

  std::vector<float> input(blockSize * numBlocks, 0.0f);
  input[0] = 1.0f;
  // Add a second impulse later
  input[blockSize * 50] = 0.5f;

  auto split = runSplit(input, ir, blockSize, numBlocks);
  auto st    = runST(input, ir, blockSize, numBlocks);

  float diff = maxDiff(split.data(), st.data(), blockSize * numBlocks);
  printf("  max diff split vs ST: %e\n", diff);
  check(diff < 1e-3f, "very long IR (2s): split matches ST");
}

// Simulates the web worker scenario: tail result arrives 1 block late
static void testSplitWithLatency()
{
  printf("\n[testSplitWithLatency]\n");

  const size_t blockSize = 128;
  const size_t irLen = blockSize * 20;
  const size_t numBlocks = 40;

  std::vector<float> ir(irLen, 0.0f);
  for (size_t i = 0; i < irLen; ++i)
    ir[i] = std::exp(-3.0f * static_cast<float>(i) / static_cast<float>(irLen));

  std::vector<float> input(blockSize * numBlocks, 0.0f);
  input[0] = 1.0f;

  auto level0IrFFT = buildLevel0IrFFT(ir, blockSize);
  ConvolutionLevel level0(0, blockSize, 0, 2);
  TailEngine tail;
  tail.loadIR(ir.data(), ir.size(), blockSize);

  // Simulate 1-block latency: tail result from block N applied at block N+1
  std::vector<float> prevTailResult(blockSize, 0.0f);
  std::vector<float> output(blockSize * numBlocks, 0.0f);

  for (size_t block = 0; block < numBlocks; ++block) {
    const float* in  = input.data() + block * blockSize;
    float*       out = output.data() + block * blockSize;

    level0.processBlock(in, level0IrFFT);

    // Apply PREVIOUS block's tail result (simulating message latency)
    const float* level0Out = level0.getResult();
    for (size_t i = 0; i < blockSize; ++i)
      out[i] = level0Out[i] + prevTailResult[i];

    // Process tail for this block (result arrives "next block")
    tail.processBlock(in);
    const float* tl = tail.getResult();
    std::copy(tl, tl + blockSize, prevTailResult.begin());
  }

  // Compare to no-latency split
  auto reference = runSplit(input, ir, blockSize, numBlocks);

  // The 1-block latency shifts the tail by 1 block (128 samples).
  // For reverb, this is inaudible. Check that the overall shape is similar
  // by comparing with a 1-block shift allowance.
  float diffShifted = 0.0f;
  for (size_t i = blockSize; i < blockSize * numBlocks; ++i)
    diffShifted = std::fmax(diffShifted,
                             std::fabs(output[i] - reference[i - blockSize]));

  // This won't be exact because level0 isn't shifted, only the tail is.
  // But the tail starts at seg 1 so it's naturally delayed. The energy
  // should be in the right ballpark.
  float totalEnergy = 0.0f;
  float diffEnergy = 0.0f;
  for (size_t i = 0; i < blockSize * numBlocks; ++i) {
    totalEnergy += reference[i] * reference[i];
    float d = output[i] - reference[i];
    diffEnergy += d * d;
  }
  float snr = 10.0f * std::log10(totalEnergy / (diffEnergy + 1e-30f));

  printf("  SNR (latency vs no-latency): %.1f dB\n", snr);
  printf("  (>20 dB is good for reverb with 1-block tail latency)\n");
  check(snr > 20.0f, "split with 1-block latency has acceptable SNR");
}

// ----------------------------------------------------------------------------

int main()
{
  printf("=== Split architecture (level0 + TailEngine) tests ===\n");

  testLongIR_Impulse();
  testLongIR_Sine();
  testVeryLongIR();
  testSplitWithLatency();

  printf("\n%d passed, %d failed\n", sTestsPassed, sTestsFailed);
  return sTestsFailed > 0 ? 1 : 0;
}
