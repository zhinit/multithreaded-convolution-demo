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

// Build an IR super-segment FFT for a given level from raw IR data
// Returns as a single-element partition vector for UPOLS API
static std::vector<std::vector<float>> buildIrSuperFFT(const std::vector<float>& ir,
                                                        size_t blockSize,
                                                        int level)
{
  size_t period = 1 << level;
  size_t superBlockSize = period * blockSize;
  size_t fftSize = 2 * superBlockSize;
  size_t irSegOffset = (1 << level) - 1; // first segment for this level

  std::vector<float> irSuper(fftSize, 0.0f);
  size_t numSegments = (ir.size() + blockSize - 1) / blockSize;
  for (size_t seg = 0; seg < period && (irSegOffset + seg) < numSegments; ++seg) {
    size_t srcOffset = (irSegOffset + seg) * blockSize;
    size_t count = std::min(blockSize, ir.size() - srcOffset);
    for (size_t i = 0; i < count; ++i)
      irSuper[seg * blockSize + i] = ir[srcOffset + i];
  }

  std::vector<float> irFFT(fftSize * 2, 0.0f);
  fft(irSuper.data(), irFFT.data(), fftSize);
  return { irFFT };
}

// Build a level-0 IR FFT (just FFT of first blockSize samples)
// Returns as a single-element partition vector for UPOLS API
static std::vector<std::vector<float>> buildLevel0IrFFT(const std::vector<float>& ir,
                                                         size_t blockSize)
{
  size_t fftSize = blockSize * 2;
  std::vector<float> slice(fftSize, 0.0f);
  size_t count = std::min(blockSize, ir.size());
  for (size_t i = 0; i < count; ++i)
    slice[i] = ir[i];
  std::vector<float> irFFT(fftSize * 2, 0.0f);
  fft(slice.data(), irFFT.data(), fftSize);
  return { irFFT };
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

// Input of [1, 0, 0, ...] with IR [0, 0.5, 0, 0.5] should produce [0, 0.5, 0, 0.5]
static void testImpulseInput()
{
  printf("\n[testImpulseInput]\n");

  const size_t blockSize = 128;
  std::vector<float> ir = { 0.0f, 0.5f, 0.0f, 0.5f };

  auto irFFT = buildLevel0IrFFT(ir, blockSize);

  ConvolutionLevel level(0, blockSize);

  // first block: impulse at sample 0
  std::vector<float> block(blockSize, 0.0f);
  block[0] = 1.0f;
  level.processBlock(block.data(), irFFT);

  const float* result = level.getResult();
  float expected[] = { 0.0f, 0.5f, 0.0f, 0.5f };
  float diff = maxDiff(result, expected, 4);
  printf("  max diff: %e\n", diff);
  check(diff < 1e-3f, "impulse input produces IR as output");
}

// Input of silence should produce silence
static void testSilentInput()
{
  printf("\n[testSilentInput]\n");

  const size_t blockSize = 128;
  std::vector<float> ir = { 1.0f, 0.5f, 0.25f };

  auto irFFT = buildLevel0IrFFT(ir, blockSize);

  ConvolutionLevel level(0, blockSize);

  std::vector<float> block(blockSize, 0.0f);
  level.processBlock(block.data(), irFFT);

  const float* result = level.getResult();
  bool silent = true;
  for (size_t i = 0; i < blockSize; ++i)
    if (std::fabs(result[i]) > 1e-5f) silent = false;

  check(silent, "silent input produces silent output");
}

// Level 1 should produce output that changes every block (not just on firing blocks)
static void testLevel1ProducesOutputEveryBlock()
{
  printf("\n[testLevel1ProducesOutputEveryBlock]\n");

  const size_t blockSize = 128;
  // IR with content in segments 1-2 (what level 1 covers)
  std::vector<float> ir(blockSize * 3, 0.0f);
  ir[blockSize] = 1.0f; // impulse in segment 1

  auto irFFT = buildIrSuperFFT(ir, blockSize, 1);

  ConvolutionLevel level(1, blockSize);

  // block 0: impulse
  std::vector<float> impulse(blockSize, 0.0f);
  impulse[0] = 1.0f;
  level.processBlock(impulse.data(), irFFT);
  // level 1 hasn't fired yet (needs 2 blocks), output is zero
  bool block0Silent = std::fabs(level.getResult()[0]) < 1e-5f;

  // block 1: silence — level 1 fires (accumulated blocks 0 and 1)
  std::vector<float> silence(blockSize, 0.0f);
  level.processBlock(silence.data(), irFFT);
  float block1Result = level.getResult()[0];

  printf("  block 0 silent: %s\n", block0Silent ? "yes" : "no");
  printf("  block 1 result[0] (expect ~1.0): %f\n", block1Result);
  check(block0Silent, "level 1 output is zero before first firing");
  check(std::fabs(block1Result - 1.0f) < 1e-3f,
        "level 1 produces correct output on firing block");
}

// After reset, silent input should produce silent output
static void testReset()
{
  printf("\n[testReset]\n");

  const size_t blockSize = 128;
  std::vector<float> ir = { 1.0f, 0.5f, 0.25f };
  auto irFFT = buildLevel0IrFFT(ir, blockSize);

  ConvolutionLevel level(0, blockSize);

  // dirty the state
  std::vector<float> block(blockSize, 1.0f);
  level.processBlock(block.data(), irFFT);

  level.reset();

  std::vector<float> silence(blockSize, 0.0f);
  level.processBlock(silence.data(), irFFT);

  const float* result = level.getResult();
  bool silent = true;
  for (size_t i = 0; i < blockSize; ++i)
    if (std::fabs(result[i]) > 1e-5f) silent = false;

  check(silent, "silent output after reset");
}

// Feed a block, then silence. The overlap tail from the convolution should
// appear in the next firing's result.
static void testOverlapTailCarriesOver()
{
  printf("\n[testOverlapTailCarriesOver]\n");

  const size_t blockSize = 128;
  std::vector<float> ir(blockSize, 1.0f);
  auto irFFT = buildLevel0IrFFT(ir, blockSize);

  ConvolutionLevel level(0, blockSize);

  // block 0: all ones
  std::vector<float> ones(blockSize, 1.0f);
  level.processBlock(ones.data(), irFFT);

  // block 1: silence — overlap from block 0 should appear
  std::vector<float> silence(blockSize, 0.0f);
  level.processBlock(silence.data(), irFFT);
  float firstSampleBlock1 = level.getResult()[0];

  printf("  result[0] on block 1 (expect > 0): %f\n", firstSampleBlock1);
  check(firstSampleBlock1 > 1e-3f, "overlap tail carries into next block");
}

// Level 0 fires every block — verify it fires on blockCount=1, 2, 3.
static void testLevel0FiresEveryBlock()
{
  printf("\n[testLevel0FiresEveryBlock]\n");

  const size_t blockSize = 128;
  std::vector<float> ir = { 1.0f };
  auto irFFT = buildLevel0IrFFT(ir, blockSize);

  ConvolutionLevel level(0, blockSize);

  std::vector<float> impulse(blockSize, 0.0f);
  impulse[0] = 1.0f;

  bool allFired = true;
  for (size_t blockCount = 0; blockCount < 4; ++blockCount) {
    level.processBlock(impulse.data(), irFFT);
    if (std::fabs(level.getResult()[0] - 1.0f) > 1e-3f)
      allFired = false;
  }

  check(allFired, "level 0 fires on every block");
}

// ----------------------------------------------------------------------------
// TailEngine tests
// ----------------------------------------------------------------------------

// IR impulse in seg 2 — Gardner TailEngine level 1 starts at segment 2
static void testTailEngineBasic()
{
  printf("\n[testTailEngineBasic]\n");

  const size_t blockSize = 128;
  // Gardner: level 0 covers segs 0-1, level 1 covers segs 2-5
  std::vector<float> ir(blockSize * 4, 0.0f);
  ir[blockSize * 2] = 1.0f; // impulse at start of seg 2

  TailEngine engine;
  engine.loadIR(ir.data(), ir.size(), blockSize);

  std::vector<float> impulse(blockSize, 0.0f);
  impulse[0] = 1.0f;
  std::vector<float> silence(blockSize, 0.0f);

  // Level 1: period=2, clearance=1 (output delayed 1 block)
  // Fire at block 1, output served starting at block 2
  engine.processBlock(impulse.data()); // block 0
  engine.processBlock(silence.data()); // block 1: fire, but output delayed
  float result1 = engine.getResult()[0];
  printf("  result[0] after block 1 (expect 0, delayed): %f\n", result1);

  engine.processBlock(silence.data()); // block 2: delayed output arrives
  float result2 = engine.getResult()[0];
  printf("  result[0] after block 2 (expect ~1.0): %f\n", result2);

  check(std::fabs(result1) < 1e-3f, "tail engine output delayed by clearance");
  check(std::fabs(result2 - 1.0f) < 1e-3f, "tail engine produces output at correct deadline");
}

// IR only in segs 0-1 — Gardner TailEngine handles segs 2+, so result should be silent.
static void testTailEngineIgnoresLevel0Segs()
{
  printf("\n[testTailEngineIgnoresLevel0Segs]\n");

  const size_t blockSize = 128;
  // IR content only in segments 0-1 (level 0's Gardner coverage)
  std::vector<float> ir(blockSize * 2, 1.0f);

  TailEngine engine;
  engine.loadIR(ir.data(), ir.size(), blockSize);

  std::vector<float> impulse(blockSize, 0.0f);
  impulse[0] = 1.0f;
  engine.processBlock(impulse.data());

  bool silent = true;
  for (size_t i = 0; i < blockSize; ++i)
    if (std::fabs(engine.getResult()[i]) > 1e-5f) silent = false;

  check(silent, "tail engine is silent when IR only has segs 0-1 content");
}

// Silent input should produce silent output.
static void testTailEngineSilentInput()
{
  printf("\n[testTailEngineSilentInput]\n");

  const size_t blockSize = 128;
  std::vector<float> ir(blockSize * 4, 0.0f);
  ir[blockSize * 2] = 1.0f; // impulse in seg 2 (Gardner tail territory)

  TailEngine engine;
  engine.loadIR(ir.data(), ir.size(), blockSize);

  std::vector<float> silence(blockSize, 0.0f);
  engine.processBlock(silence.data());

  bool silent = true;
  for (size_t i = 0; i < blockSize; ++i)
    if (std::fabs(engine.getResult()[i]) > 1e-5f) silent = false;

  check(silent, "tail engine is silent with silent input");
}

// After reset, silent input should produce silent output.
static void testTailEngineReset()
{
  printf("\n[testTailEngineReset]\n");

  const size_t blockSize = 128;
  std::vector<float> ir(blockSize * 4, 0.0f);
  ir[blockSize * 2] = 1.0f; // impulse in seg 2

  TailEngine engine;
  engine.loadIR(ir.data(), ir.size(), blockSize);

  // dirty the state
  std::vector<float> impulse(blockSize, 0.0f);
  impulse[0] = 1.0f;
  engine.processBlock(impulse.data());
  engine.processBlock(impulse.data());
  engine.processBlock(impulse.data());

  engine.reset();

  std::vector<float> silence(blockSize, 0.0f);
  engine.processBlock(silence.data());

  bool silent = true;
  for (size_t i = 0; i < blockSize; ++i)
    if (std::fabs(engine.getResult()[i]) > 1e-5f) silent = false;

  check(silent, "tail engine is silent after reset");
}

// ----------------------------------------------------------------------------

int main()
{
  printf("=== convolution_mt tests ===\n");

  testImpulseInput();
  testSilentInput();
  testLevel1ProducesOutputEveryBlock();
  testReset();
  testOverlapTailCarriesOver();
  testLevel0FiresEveryBlock();

  testTailEngineBasic();
  testTailEngineIgnoresLevel0Segs();
  testTailEngineSilentInput();
  testTailEngineReset();

  printf("\n%d passed, %d failed\n", sTestsPassed, sTestsFailed);
  return sTestsFailed > 0 ? 1 : 0;
}
