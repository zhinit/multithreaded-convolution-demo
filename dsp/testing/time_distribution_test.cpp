#include "../convolution_mine_st.h"
#include "../convolution_mt.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

static int sTestsPassed = 0;
static int sTestsFailed = 0;

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (cond) {                                                                \
      printf("  PASS: %s\n", msg);                                            \
      sTestsPassed++;                                                          \
    } else {                                                                   \
      printf("  FAIL: %s\n", msg);                                            \
      sTestsFailed++;                                                          \
    }                                                                          \
  } while (0)

// Build level 0 IR FFT (2 partitions covering IR[0..2B))
static std::vector<std::vector<float>>
buildLevel0IrFFT(const std::vector<float>& ir, size_t blockSize)
{
  size_t fftSize = 2 * blockSize;
  std::vector<std::vector<float>> partitions;

  for (int p = 0; p < 2; ++p) {
    std::vector<float> irPart(fftSize, 0.0f);
    size_t srcOff = p * blockSize;
    size_t count = std::min(blockSize, ir.size() > srcOff ? ir.size() - srcOff : (size_t)0);
    for (size_t i = 0; i < count; ++i)
      irPart[i] = ir[srcOff + i];

    std::vector<float> partFFT(fftSize * 2, 0.0f);
    fft(irPart.data(), partFFT.data(), fftSize);
    partitions.push_back(std::move(partFFT));
  }
  return partitions;
}

// Run split (level0 + TailEngine) convolution
static std::vector<float>
runSplit(const std::vector<float>& input, const std::vector<float>& ir,
         size_t blockSize, size_t numBlocks)
{
  auto level0FFT = buildLevel0IrFFT(ir, blockSize);
  ConvolutionLevel level0(0, blockSize, 0, 2);
  TailEngine tail;
  tail.loadIR(ir.data(), ir.size(), blockSize);

  std::vector<float> output(blockSize * numBlocks, 0.0f);
  for (size_t b = 0; b < numBlocks; ++b) {
    const float* in = input.data() + b * blockSize;
    float* out = output.data() + b * blockSize;

    level0.processBlock(in, level0FFT);
    tail.processBlock(in);

    const float* level0Out = level0.getResult();
    const float* tl = tail.getResult();
    for (size_t i = 0; i < blockSize; ++i)
      out[i] = level0Out[i] + tl[i];
  }
  return output;
}

// --- Test 1: MT vs ST correctness with pipeline active ---
static void
testPipelinedMTvsST()
{
  printf("\n[testPipelinedMTvsST]\n");

  const size_t blockSize = 128;
  const size_t irLen = blockSize * 40; // ~5000 samples, several levels
  const size_t numBlocks = 80;

  // Exponential decay IR
  std::vector<float> ir(irLen);
  for (size_t i = 0; i < irLen; ++i)
    ir[i] = std::exp(-3.0f * (float)i / (float)irLen);

  // Sine input
  std::vector<float> input(blockSize * numBlocks, 0.0f);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = std::sin(2.0f * 3.14159265f * 440.0f * (float)i / 44100.0f);

  // Run split MT (with pipeline)
  auto mtOut = runSplit(input, ir, blockSize, numBlocks);

  // Run single-threaded reference
  ConvolutionEngine st;
  st.loadIR(ir.data(), ir.size());
  std::vector<float> stOut(blockSize * numBlocks, 0.0f);
  for (size_t b = 0; b < numBlocks; ++b) {
    st.process(input.data() + b * blockSize,
               stOut.data() + b * blockSize, blockSize);
  }

  float maxDiff = 0.0f;
  for (size_t i = 0; i < mtOut.size(); ++i)
    maxDiff = std::fmax(maxDiff, std::fabs(mtOut[i] - stOut[i]));

  printf("  max diff pipelined MT vs ST: %e\n", maxDiff);
  CHECK(maxDiff < 1e-3f, "pipelined MT matches ST reference");
}

// --- Test 2: MT vs ST with long IR (2s) ---
static void
testPipelinedLongIR()
{
  printf("\n[testPipelinedLongIR]\n");

  const size_t blockSize = 128;
  const size_t irLen = 44100 * 2; // 2 seconds at 44.1kHz
  const size_t numBlocks = 800;

  std::vector<float> ir(irLen);
  for (size_t i = 0; i < irLen; ++i)
    ir[i] = std::exp(-4.0f * (float)i / (float)irLen) *
            std::sin(2.0f * 3.14159265f * 100.0f * (float)i / 44100.0f);

  // Impulse input
  std::vector<float> input(blockSize * numBlocks, 0.0f);
  input[0] = 1.0f;

  auto mtOut = runSplit(input, ir, blockSize, numBlocks);

  ConvolutionEngine st;
  st.loadIR(ir.data(), ir.size());
  std::vector<float> stOut(blockSize * numBlocks, 0.0f);
  for (size_t b = 0; b < numBlocks; ++b) {
    st.process(input.data() + b * blockSize,
               stOut.data() + b * blockSize, blockSize);
  }

  float maxDiff = 0.0f;
  for (size_t i = 0; i < mtOut.size(); ++i)
    maxDiff = std::fmax(maxDiff, std::fabs(mtOut[i] - stOut[i]));

  printf("  max diff pipelined MT vs ST (2s IR): %e\n", maxDiff);
  CHECK(maxDiff < 1e-2f, "pipelined MT matches ST for 2s IR");
}

// --- Test 3: Time distribution — max/avg ratio ---
static void
testTimeDistribution()
{
  printf("\n[testTimeDistribution]\n");

  const size_t blockSize = 128;
  const size_t sampleRate = 44100;
  const size_t irLen = sampleRate * 30; // 30 seconds
  const size_t numBlocks = 2000;

  // Generate 30s IR (exponential decay)
  std::vector<float> ir(irLen);
  for (size_t i = 0; i < irLen; ++i)
    ir[i] = std::exp(-2.0f * (float)i / (float)irLen);

  // Build level 0
  auto level0FFT = buildLevel0IrFFT(ir, blockSize);
  ConvolutionLevel level0(0, blockSize, 0, 2);

  // Build tail engine
  TailEngine tail;
  tail.loadIR(ir.data(), ir.size(), blockSize);

  // White noise input
  std::vector<float> input(blockSize * numBlocks);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = (float)(rand() % 10000 - 5000) / 5000.0f;

  // Measure per-block timing
  std::vector<double> blockTimes(numBlocks);
  std::vector<float> output(blockSize, 0.0f);

  for (size_t b = 0; b < numBlocks; ++b) {
    const float* in = input.data() + b * blockSize;

    auto t0 = std::chrono::high_resolution_clock::now();

    level0.processBlock(in, level0FFT);
    tail.processBlock(in);

    // Mix
    const float* level0Out = level0.getResult();
    const float* tl = tail.getResult();
    for (size_t i = 0; i < blockSize; ++i)
      output[i] = level0Out[i] + tl[i];

    auto t1 = std::chrono::high_resolution_clock::now();
    blockTimes[b] =
      std::chrono::duration<double, std::micro>(t1 - t0).count();
  }

  // Skip first 50 blocks (warmup / initial pipeline fills)
  size_t start = 50;
  double maxTime = 0.0;
  double sumTime = 0.0;
  size_t count = numBlocks - start;

  for (size_t b = start; b < numBlocks; ++b) {
    if (blockTimes[b] > maxTime)
      maxTime = blockTimes[b];
    sumTime += blockTimes[b];
  }
  double avgTime = sumTime / (double)count;
  double ratio = maxTime / avgTime;

  printf("  IR length: %zu samples (%.1f seconds)\n", irLen,
         (float)irLen / sampleRate);
  printf("  Blocks processed: %zu (skipped first %zu for warmup)\n",
         count, start);
  printf("  Avg block time: %.1f us\n", avgTime);
  printf("  Max block time: %.1f us\n", maxTime);
  printf("  Max/avg ratio: %.2f\n", ratio);

  // Show top 10 slowest blocks
  std::vector<std::pair<double, size_t>> sorted;
  for (size_t b = start; b < numBlocks; ++b)
    sorted.push_back({blockTimes[b], b});
  std::sort(sorted.begin(), sorted.end(),
            [](auto& a, auto& b) { return a.first > b.first; });
  printf("  Top 10 slowest blocks:\n");
  for (int i = 0; i < 10 && i < (int)sorted.size(); ++i)
    printf("    block %zu: %.1f us\n", sorted[i].second, sorted[i].first);

  CHECK(ratio < 3.0, "max/avg block time ratio < 3.0 (time is distributed)");
}

int
main()
{
  printf("=== Time Distribution tests ===\n");

  testPipelinedMTvsST();
  testPipelinedLongIR();
  testTimeDistribution();

  printf("\n%d passed, %d failed\n", sTestsPassed, sTestsFailed);
  return sTestsFailed > 0 ? 1 : 0;
}
