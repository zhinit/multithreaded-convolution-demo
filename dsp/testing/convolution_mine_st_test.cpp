#include "../convolution_mine_st.h"
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

// Returns true if all values in buf are within tolerance of expected
static bool allClose(const float* buf, size_t len, float expected, float tol = 1e-3f)
{
  for (size_t i = 0; i < len; ++i) {
    if (std::fabs(buf[i] - expected) > tol)
      return false;
  }
  return true;
}

// Returns max absolute difference between two buffers
static float maxDiff(const float* a, const float* b, size_t len)
{
  float max = 0.0f;
  for (size_t i = 0; i < len; ++i)
    max = std::fmax(max, std::fabs(a[i] - b[i]));
  return max;
}

// Naive time-domain convolution for reference
static std::vector<float> naiveConvolve(const std::vector<float>& input,
                                        const std::vector<float>& ir)
{
  size_t outLen = input.size() + ir.size() - 1;
  std::vector<float> out(outLen, 0.0f);
  for (size_t i = 0; i < input.size(); ++i)
    for (size_t j = 0; j < ir.size(); ++j)
      out[i + j] += input[i] * ir[j];
  return out;
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

// Convolving with a unit impulse [1, 0, 0, ...] should return the input.
static void testImpulseIR()
{
  printf("\n[testImpulseIR]\n");

  const size_t blockSize = 128;
  const size_t numBlocks = 4;
  const size_t irLen = 384; // one segment

  // IR = unit impulse
  std::vector<float> ir(irLen, 0.0f);
  ir[0] = 1.0f;

  ConvolutionEngine engine;
  engine.loadIR(ir.data(), irLen);

  // Input: ramp signal across 4 blocks
  std::vector<float> input(blockSize * numBlocks);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<float>(i) / static_cast<float>(input.size());

  std::vector<float> output(blockSize * numBlocks);

  for (size_t block = 0; block < numBlocks; ++block) {
    const float* in = input.data() + block * blockSize;
    float* out = output.data() + block * blockSize;
    engine.process(in, out, blockSize);
  }

  // Output should match input (within tolerance)
  float diff = maxDiff(input.data(), output.data(), blockSize * numBlocks);
  printf("  max diff from input: %e\n", diff);
  check(diff < 1e-3f, "impulse IR output matches input");
}

// Convolving with an IR of all zeros should produce silence.
static void testZeroIR()
{
  printf("\n[testZeroIR]\n");

  const size_t blockSize = 128;
  const size_t irLen = 384;

  std::vector<float> ir(irLen, 0.0f);

  ConvolutionEngine engine;
  engine.loadIR(ir.data(), irLen);

  std::vector<float> input(blockSize);
  for (size_t i = 0; i < blockSize; ++i)
    input[i] = 1.0f;

  std::vector<float> output(blockSize);
  engine.process(input.data(), output.data(), blockSize);

  check(allClose(output.data(), blockSize, 0.0f, 1e-5f), "zero IR produces silence");
}

// Process silence — output should remain silent even with a loaded IR.
static void testSilentInput()
{
  printf("\n[testSilentInput]\n");

  const size_t blockSize = 128;
  const size_t irLen = 384;

  std::vector<float> ir(irLen, 0.0f);
  ir[0] = 1.0f;

  ConvolutionEngine engine;
  engine.loadIR(ir.data(), irLen);

  std::vector<float> input(blockSize, 0.0f);
  std::vector<float> output(blockSize, 0.0f);
  engine.process(input.data(), output.data(), blockSize);

  check(allClose(output.data(), blockSize, 0.0f, 1e-5f), "silent input produces silent output");
}

// Compare output against naive time-domain convolution for a short IR and
// a few blocks of input. Tests multi-block correctness and overlap-add.
static void testAgainstNaive()
{
  printf("\n[testAgainstNaive]\n");

  const size_t blockSize = 128;
  const size_t numBlocks = 6;
  const size_t totalSamples = blockSize * numBlocks;

  // Short IR (fits in one segment)
  std::vector<float> ir = { 1.0f, 0.5f, 0.25f, 0.125f };
  const size_t irLen = ir.size();

  ConvolutionEngine engine;
  engine.loadIR(ir.data(), irLen);

  // Sine wave input
  std::vector<float> input(totalSamples);
  for (size_t i = 0; i < totalSamples; ++i)
    input[i] = std::sin(2.0f * M_PI * 440.0f * i / 44100.0f);

  // Naive reference (only compute as many output samples as we'll check)
  std::vector<float> reference = naiveConvolve(input, ir);

  // Process block by block
  std::vector<float> output(totalSamples);
  for (size_t block = 0; block < numBlocks; ++block) {
    const float* in = input.data() + block * blockSize;
    float* out = output.data() + block * blockSize;
    engine.process(in, out, blockSize);
  }

  float diff = maxDiff(output.data(), reference.data(), totalSamples);
  printf("  max diff from naive reference: %e\n", diff);
  check(diff < 1e-3f, "matches naive convolution (single segment IR)");
}

// Same as above but with an IR that spans multiple segments, exercising
// the circular input history buffer.
static void testMultiSegmentIR()
{
  printf("\n[testMultiSegmentIR]\n");

  const size_t blockSize = 128;
  const size_t numBlocks = 16;
  const size_t totalSamples = blockSize * numBlocks;

  // IR long enough to span 3 segments (segmentSize = 384, so 3 * 384 = 1152)
  const size_t irLen = 1152;
  std::vector<float> ir(irLen, 0.0f);
  // Exponential decay
  for (size_t i = 0; i < irLen; ++i)
    ir[i] = std::exp(-3.0f * static_cast<float>(i) / static_cast<float>(irLen));

  ConvolutionEngine engine;
  engine.loadIR(ir.data(), irLen);

  std::vector<float> input(totalSamples, 0.0f);
  input[0] = 1.0f; // single impulse in, so output should equal IR

  std::vector<float> reference = naiveConvolve(input, ir);

  std::vector<float> output(totalSamples);
  for (size_t block = 0; block < numBlocks; ++block) {
    const float* in = input.data() + block * blockSize;
    float* out = output.data() + block * blockSize;
    engine.process(in, out, blockSize);
  }

  float diff = maxDiff(output.data(), reference.data(), totalSamples);
  printf("  max diff from naive reference: %e\n", diff);
  check(diff < 1e-3f, "matches naive convolution (multi-segment IR)");
}

// After reset(), engine should behave as if freshly loaded.
static void testReset()
{
  printf("\n[testReset]\n");

  const size_t blockSize = 128;
  const size_t irLen = 384;

  std::vector<float> ir(irLen, 0.0f);
  ir[0] = 1.0f;

  ConvolutionEngine engine;
  engine.loadIR(ir.data(), irLen);

  // Run a block of non-silent input to dirty the state
  std::vector<float> input(blockSize, 1.0f);
  std::vector<float> output(blockSize);
  engine.process(input.data(), output.data(), blockSize);

  engine.reset();

  // After reset, silent input should yield silent output
  std::vector<float> silentInput(blockSize, 0.0f);
  std::vector<float> silentOutput(blockSize, 0.0f);
  engine.process(silentInput.data(), silentOutput.data(), blockSize);

  check(allClose(silentOutput.data(), blockSize, 0.0f, 1e-5f),
        "silent output after reset");
}

// ----------------------------------------------------------------------------

int main()
{
  printf("=== convolution_mine_st tests ===\n");

  testImpulseIR();
  testZeroIR();
  testSilentInput();
  testAgainstNaive();
  testMultiSegmentIR();
  testReset();

  printf("\n%d passed, %d failed\n", sTestsPassed, sTestsFailed);
  return sTestsFailed > 0 ? 1 : 0;
}
