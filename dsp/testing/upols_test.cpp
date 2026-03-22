#include "../convolution_mine_st.h"
#include "../convolution_mt.h"
#include "../fft.h"
#include <cmath>
#include <cstdio>
#include <vector>

// ----------------------------------------------------------------------------
// Minimal test harness
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

// Build a single-partition IR FFT for a ConvolutionLevel
// irSegData: the raw IR samples for this level's segment
// fftSize: the level's FFT size
static std::vector<float> buildPartitionFFT(const float* data, size_t dataLen,
                                             size_t fftSize)
{
  std::vector<float> padded(fftSize, 0.0f);
  for (size_t i = 0; i < dataLen && i < fftSize; ++i)
    padded[i] = data[i];
  std::vector<float> result(fftSize * 2, 0.0f);
  fft(padded.data(), result.data(), fftSize);
  return result;
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

// P=1 with new API must match old single-partition behavior
static void testSinglePartitionMatchesOld()
{
  printf("\n[testSinglePartitionMatchesOld]\n");

  const size_t blockSize = 128;
  std::vector<float> ir = { 0.0f, 0.5f, 0.0f, 0.5f };

  // Build single-partition FFT (level 0, P=1)
  size_t fftSize = blockSize * 2;
  std::vector<float> slice(fftSize, 0.0f);
  for (size_t i = 0; i < ir.size(); ++i)
    slice[i] = ir[i];
  auto partFFT = buildPartitionFFT(slice.data(), fftSize, fftSize);
  std::vector<std::vector<float>> partitions = { partFFT };

  ConvolutionLevel level(0, blockSize, 0, 1);

  std::vector<float> block(blockSize, 0.0f);
  block[0] = 1.0f;
  level.processBlock(block.data(), partitions);

  const float* result = level.getResult();
  float expected[] = { 0.0f, 0.5f, 0.0f, 0.5f };
  float diff = maxDiff(result, expected, 4);
  printf("  max diff: %e\n", diff);
  check(diff < 1e-3f, "P=1 impulse matches expected output");
}

// P=2 UPOLS level 0: IR covering 2 blocks, split into 2 partitions
// With an impulse input, output should match the full IR
static void testTwoPartitionImpulse()
{
  printf("\n[testTwoPartitionImpulse]\n");

  const size_t blockSize = 128;
  const size_t fftSize = blockSize * 2;

  // IR: 2 blocks long. Partition 0 = first block, partition 1 = second block
  std::vector<float> ir(blockSize * 2, 0.0f);
  ir[0] = 1.0f;
  ir[blockSize] = 0.5f;
  ir[blockSize + 1] = 0.25f;

  // Build 2 sub-filter FFTs
  auto p0 = buildPartitionFFT(ir.data(), blockSize, fftSize);
  auto p1 = buildPartitionFFT(ir.data() + blockSize, blockSize, fftSize);
  std::vector<std::vector<float>> partitions = { p0, p1 };

  ConvolutionLevel level(0, blockSize, 0, 2);

  // Block 0: impulse
  std::vector<float> impulse(blockSize, 0.0f);
  impulse[0] = 1.0f;
  level.processBlock(impulse.data(), partitions);

  // Block 0 result should have partition 0's contribution: ir[0]=1.0
  float diff0 = std::fabs(level.getResult()[0] - 1.0f);
  printf("  block 0 result[0] (expect 1.0): %f\n", level.getResult()[0]);

  // Block 1: silence — partition 1 kicks in (previous input × partition 1)
  std::vector<float> silence(blockSize, 0.0f);
  level.processBlock(silence.data(), partitions);

  float result1_0 = level.getResult()[0];
  float result1_1 = level.getResult()[1];
  printf("  block 1 result[0] (expect 0.5): %f\n", result1_0);
  printf("  block 1 result[1] (expect 0.25): %f\n", result1_1);

  check(diff0 < 1e-3f, "P=2 block 0: partition 0 contributes correctly");
  check(std::fabs(result1_0 - 0.5f) < 1e-3f, "P=2 block 1: partition 1 result[0]");
  check(std::fabs(result1_1 - 0.25f) < 1e-3f, "P=2 block 1: partition 1 result[1]");
}

// P=2 UPOLS level 1 (period=2): verify correct convolution with 2 partitions
// Each partition covers period*blockSize = 256 samples
static void testTwoPartitionLevel1()
{
  printf("\n[testTwoPartitionLevel1]\n");

  const size_t blockSize = 128;
  const size_t period = 2;
  const size_t superBlockSize = period * blockSize; // 256
  const size_t fftSize = 2 * superBlockSize;        // 512

  // IR: 4 blocks (512 samples). Split into 2 partitions of 256 samples each
  std::vector<float> ir(superBlockSize * 2, 0.0f);
  ir[0] = 1.0f;             // partition 0, sample 0
  ir[superBlockSize] = 0.5f; // partition 1, sample 0

  auto p0 = buildPartitionFFT(ir.data(), superBlockSize, fftSize);
  auto p1 = buildPartitionFFT(ir.data() + superBlockSize, superBlockSize, fftSize);
  std::vector<std::vector<float>> partitions = { p0, p1 };

  ConvolutionLevel level(1, blockSize, 0, 2);

  // Process 8 blocks with impulse at block 0
  std::vector<float> impulse(blockSize, 0.0f);
  impulse[0] = 1.0f;
  std::vector<float> silence(blockSize, 0.0f);

  // Block 0: accumulate (level 1 hasn't fired yet)
  level.processBlock(impulse.data(), partitions);
  bool block0Silent = std::fabs(level.getResult()[0]) < 1e-5f;

  // Block 1: fire! (accumulated blocks 0-1). Partition 0 contributes.
  level.processBlock(silence.data(), partitions);
  float block1_0 = level.getResult()[0];
  printf("  block 1 result[0] (expect 1.0): %f\n", block1_0);

  // Blocks 2-3: serve remaining output from first firing
  level.processBlock(silence.data(), partitions);
  level.processBlock(silence.data(), partitions);
  // Block 3 fires again. Now partition 1 kicks in (previous input × H1)
  float block3_0 = level.getResult()[0];
  printf("  block 3 result[0] (expect 0.5): %f\n", block3_0);

  check(block0Silent, "P=2 level 1: silent before first fire");
  check(std::fabs(block1_0 - 1.0f) < 1e-3f, "P=2 level 1: partition 0 fires correctly");
  check(std::fabs(block3_0 - 0.5f) < 1e-3f, "P=2 level 1: partition 1 fires on second fire");
}

// Compare P=2 UPOLS against single-threaded reference
// Build a level 0 with P=2 covering IR[0..2B), verify against ST reference
static void testUPOLSvsST()
{
  printf("\n[testUPOLSvsST]\n");

  const size_t blockSize = 128;
  const size_t fftSize = blockSize * 2;
  const size_t numBlocks = 16;

  // IR: 2 blocks long, exponential decay
  std::vector<float> ir(blockSize * 2, 0.0f);
  for (size_t i = 0; i < ir.size(); ++i)
    ir[i] = std::exp(-3.0f * static_cast<float>(i) / static_cast<float>(ir.size()));

  // UPOLS P=2: partition 0 = IR[0..B), partition 1 = IR[B..2B)
  auto p0 = buildPartitionFFT(ir.data(), blockSize, fftSize);
  auto p1 = buildPartitionFFT(ir.data() + blockSize, blockSize, fftSize);
  std::vector<std::vector<float>> partitions = { p0, p1 };

  ConvolutionLevel level(0, blockSize, 0, 2);

  // ST reference
  ConvolutionEngine stEngine;
  stEngine.loadIR(ir.data(), ir.size());

  // Sine input
  std::vector<float> input(blockSize * numBlocks);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = std::sin(2.0f * M_PI * 440.0f * i / 44100.0f);

  std::vector<float> upolsOutput(blockSize * numBlocks, 0.0f);
  std::vector<float> stOutput(blockSize * numBlocks, 0.0f);

  for (size_t block = 0; block < numBlocks; ++block) {
    const float* in = input.data() + block * blockSize;

    level.processBlock(in, partitions);
    std::copy(level.getResult(), level.getResult() + blockSize,
              upolsOutput.data() + block * blockSize);

    stEngine.process(in, stOutput.data() + block * blockSize, blockSize);
  }

  float diff = maxDiff(upolsOutput.data(), stOutput.data(), blockSize * numBlocks);
  printf("  max diff UPOLS vs ST: %e\n", diff);
  check(diff < 1e-3f, "P=2 UPOLS matches ST reference for 2-block IR");
}

// Test P=2 with longer IR and more blocks, compare against ST
static void testUPOLSvsSTLongerIR()
{
  printf("\n[testUPOLSvsSTLongerIR]\n");

  const size_t blockSize = 128;
  const size_t fftSize = blockSize * 2;
  const size_t numBlocks = 32;

  // IR: 2 blocks, random-ish pattern
  std::vector<float> ir(blockSize * 2, 0.0f);
  for (size_t i = 0; i < ir.size(); ++i)
    ir[i] = std::sin(0.1f * i) * std::exp(-2.0f * static_cast<float>(i) / ir.size());

  auto p0 = buildPartitionFFT(ir.data(), blockSize, fftSize);
  auto p1 = buildPartitionFFT(ir.data() + blockSize, blockSize, fftSize);
  std::vector<std::vector<float>> partitions = { p0, p1 };

  ConvolutionLevel level(0, blockSize, 0, 2);

  ConvolutionEngine stEngine;
  stEngine.loadIR(ir.data(), ir.size());

  // Impulse + noise-like input
  std::vector<float> input(blockSize * numBlocks, 0.0f);
  input[0] = 1.0f;
  for (size_t i = 1; i < input.size(); ++i)
    input[i] = std::sin(i * 0.37f) * 0.3f;

  std::vector<float> upolsOutput(blockSize * numBlocks, 0.0f);
  std::vector<float> stOutput(blockSize * numBlocks, 0.0f);

  for (size_t block = 0; block < numBlocks; ++block) {
    const float* in = input.data() + block * blockSize;
    level.processBlock(in, partitions);
    std::copy(level.getResult(), level.getResult() + blockSize,
              upolsOutput.data() + block * blockSize);
    stEngine.process(in, stOutput.data() + block * blockSize, blockSize);
  }

  float diff = maxDiff(upolsOutput.data(), stOutput.data(), blockSize * numBlocks);
  printf("  max diff UPOLS vs ST: %e\n", diff);
  check(diff < 1e-3f, "P=2 UPOLS matches ST for mixed input");
}

// Verify reset clears spectra ring
static void testUPOLSReset()
{
  printf("\n[testUPOLSReset]\n");

  const size_t blockSize = 128;
  const size_t fftSize = blockSize * 2;

  std::vector<float> ir(blockSize * 2, 0.0f);
  ir[0] = 1.0f;
  ir[blockSize] = 0.5f;

  auto p0 = buildPartitionFFT(ir.data(), blockSize, fftSize);
  auto p1 = buildPartitionFFT(ir.data() + blockSize, blockSize, fftSize);
  std::vector<std::vector<float>> partitions = { p0, p1 };

  ConvolutionLevel level(0, blockSize, 0, 2);

  // Dirty the state
  std::vector<float> impulse(blockSize, 0.0f);
  impulse[0] = 1.0f;
  level.processBlock(impulse.data(), partitions);
  level.processBlock(impulse.data(), partitions);

  level.reset();

  // After reset, silence in should produce silence out
  std::vector<float> silence(blockSize, 0.0f);
  level.processBlock(silence.data(), partitions);

  bool silent = true;
  for (size_t i = 0; i < blockSize; ++i)
    if (std::fabs(level.getResult()[i]) > 1e-5f) silent = false;

  check(silent, "P=2 UPOLS: silent after reset");
}

// ----------------------------------------------------------------------------

int main()
{
  printf("=== UPOLS (multi-partition ConvolutionLevel) tests ===\n");

  testSinglePartitionMatchesOld();
  testTwoPartitionImpulse();
  testTwoPartitionLevel1();
  testUPOLSvsST();
  testUPOLSvsSTLongerIR();
  testUPOLSReset();

  printf("\n%d passed, %d failed\n", sTestsPassed, sTestsFailed);
  return sTestsFailed > 0 ? 1 : 0;
}
