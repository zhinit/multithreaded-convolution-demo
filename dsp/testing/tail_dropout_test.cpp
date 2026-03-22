#include "../convolution_mine_st.h"
#include "../convolution_mt.h"
#include "../fft.h"
#include <cmath>
#include <cstdio>
#include <vector>

static int sTestsPassed = 0;
static int sTestsFailed = 0;

static void
check(bool condition, const char* label)
{
  if (condition) {
    printf("  PASS: %s\n", label);
    ++sTestsPassed;
  } else {
    printf("  FAIL: %s\n", label);
    ++sTestsFailed;
  }
}

// Build level-0 IR FFTs (2 partitions covering IR[0..2B))
static std::vector<std::vector<float>>
buildLevel0IrFFT(const std::vector<float>& ir, size_t blockSize)
{
  size_t fftSize = blockSize * 2;
  std::vector<std::vector<float>> partitions;
  for (size_t p = 0; p < 2; ++p) {
    std::vector<float> slice(fftSize, 0.0f);
    size_t offset = p * blockSize;
    size_t count =
      (offset < ir.size()) ? std::min(blockSize, ir.size() - offset) : 0;
    for (size_t i = 0; i < count; ++i)
      slice[i] = ir[offset + i];
    std::vector<float> irFFT(fftSize * 2, 0.0f);
    fft(slice.data(), irFFT.data(), fftSize);
    partitions.push_back(std::move(irFFT));
  }
  return partitions;
}

// Compute block energy (sum of squares)
static float
blockEnergy(const float* data, size_t len)
{
  float sum = 0.0f;
  for (size_t i = 0; i < len; ++i)
    sum += data[i] * data[i];
  return sum;
}

// Run level0 + TailEngine, return per-block energies
static std::vector<float>
runAndCollectEnergies(const std::vector<float>& ir,
                      size_t blockSize,
                      size_t numBlocks)
{
  auto level0IrFFT = buildLevel0IrFFT(ir, blockSize);

  ConvolutionLevel level0(0, blockSize, 0, 2);
  TailEngine tail;
  tail.loadIR(ir.data(), ir.size(), blockSize);

  std::vector<float> inBuf(blockSize, 0.0f);
  std::vector<float> outBuf(blockSize, 0.0f);
  std::vector<float> energies(numBlocks, 0.0f);

  for (size_t block = 0; block < numBlocks; ++block) {
    // Impulse at block 0
    std::fill(inBuf.begin(), inBuf.end(), 0.0f);
    if (block == 0)
      inBuf[0] = 1.0f;

    level0.processBlock(inBuf.data(), level0IrFFT);
    tail.processBlock(inBuf.data());

    const float* level0Out = level0.getResult();
    const float* tl = tail.getResult();
    for (size_t i = 0; i < blockSize; ++i)
      outBuf[i] = level0Out[i] + tl[i];

    energies[block] = blockEnergy(outBuf.data(), blockSize);
  }

  return energies;
}

// Also run ST reference for comparison
static std::vector<float>
runSTEnergies(const std::vector<float>& ir,
              size_t blockSize,
              size_t numBlocks)
{
  ConvolutionEngine engine;
  engine.loadIR(ir.data(), ir.size());

  std::vector<float> inBuf(blockSize, 0.0f);
  std::vector<float> outBuf(blockSize, 0.0f);
  std::vector<float> energies(numBlocks, 0.0f);

  for (size_t block = 0; block < numBlocks; ++block) {
    std::fill(inBuf.begin(), inBuf.end(), 0.0f);
    if (block == 0)
      inBuf[0] = 1.0f;

    engine.process(inBuf.data(), outBuf.data(), blockSize);
    energies[block] = blockEnergy(outBuf.data(), blockSize);
  }

  return energies;
}

// Test: scan for dropout gaps in the tail
static void
testTailDropout()
{
  printf("\n[testTailDropout] 30s IR impulse response — scanning for gaps\n");

  const size_t blockSize = 128;
  const size_t sampleRate = 48000;
  const size_t irLenSamples = 30 * sampleRate; // 30 seconds
  const size_t numSegments = (irLenSamples + blockSize - 1) / blockSize;
  const size_t numBlocks = numSegments + 256; // extra blocks for tail

  printf("  IR length: %zu samples (%zu segments)\n", irLenSamples, numSegments);
  printf("  Processing %zu blocks (%.1fs)\n",
         numBlocks,
         (float)(numBlocks * blockSize) / sampleRate);

  // Create IR: slow exponential decay (like a real reverb tail)
  std::vector<float> ir(irLenSamples);
  for (size_t i = 0; i < irLenSamples; ++i)
    ir[i] = std::exp(-3.0f * (float)i / (float)irLenSamples);

  printf("  Running MT (level0 + TailEngine)...\n");
  auto mtEnergies = runAndCollectEnergies(ir, blockSize, numBlocks);

  printf("  Running ST reference...\n");
  auto stEnergies = runSTEnergies(ir, blockSize, numBlocks);

  // Find the last block with significant energy in ST (ground truth)
  float peakST = 0.0f;
  for (auto e : stEnergies)
    peakST = std::fmax(peakST, e);
  float threshold = peakST * 1e-8f; // -80dB below peak

  size_t lastActiveBlockST = 0;
  for (size_t i = 0; i < numBlocks; ++i) {
    if (stEnergies[i] > threshold)
      lastActiveBlockST = i;
  }

  printf("  ST tail ends at block %zu (%.2fs)\n",
         lastActiveBlockST,
         (float)(lastActiveBlockST * blockSize) / sampleRate);

  // Scan MT output for gaps: blocks where ST has energy but MT doesn't
  bool inGap = false;
  size_t gapStart = 0;
  int gapCount = 0;

  for (size_t i = 0; i <= lastActiveBlockST; ++i) {
    bool stHasEnergy = stEnergies[i] > threshold;
    bool mtHasEnergy = mtEnergies[i] > threshold;

    if (stHasEnergy && !mtHasEnergy) {
      if (!inGap) {
        gapStart = i;
        inGap = true;
      }
    } else {
      if (inGap) {
        float startSec = (float)(gapStart * blockSize) / sampleRate;
        float endSec = (float)(i * blockSize) / sampleRate;
        printf("  GAP: blocks %zu-%zu (%.2fs - %.2fs) [%zu blocks]\n",
               gapStart,
               i - 1,
               startSec,
               endSec,
               i - gapStart);
        ++gapCount;
        inGap = false;
      }
    }
  }

  if (inGap) {
    float startSec = (float)(gapStart * blockSize) / sampleRate;
    float endSec = (float)((lastActiveBlockST + 1) * blockSize) / sampleRate;
    printf("  GAP: blocks %zu-%zu (%.2fs - %.2fs) [%zu blocks]\n",
           gapStart,
           lastActiveBlockST,
           startSec,
           endSec,
           lastActiveBlockST - gapStart + 1);
    ++gapCount;
  }

  printf("  Total gaps found: %d\n", gapCount);
  check(gapCount == 0, "no dropout gaps in 30s IR tail");

  // Also report where MT has significant energy beyond ST tail end
  size_t lastActiveBlockMT = 0;
  for (size_t i = 0; i < numBlocks; ++i) {
    if (mtEnergies[i] > threshold)
      lastActiveBlockMT = i;
  }

  printf("  MT tail ends at block %zu (%.2fs)\n",
         lastActiveBlockMT,
         (float)(lastActiveBlockMT * blockSize) / sampleRate);

  if (lastActiveBlockMT > lastActiveBlockST + 10) {
    printf("  WARNING: MT tail extends %.2fs beyond ST tail\n",
           (float)((lastActiveBlockMT - lastActiveBlockST) * blockSize) /
             sampleRate);
  }
}

// Test: compare MT vs ST sample-by-sample for a long IR to find where
// divergence starts
static void
testMTvsSTLongIR()
{
  printf("\n[testMTvsSTLongIR] Comparing MT vs ST output for 30s IR\n");

  const size_t blockSize = 128;
  const size_t sampleRate = 48000;
  const size_t irLenSamples = 30 * sampleRate;
  const size_t numSegments = (irLenSamples + blockSize - 1) / blockSize;
  const size_t numBlocks = numSegments + 256;

  std::vector<float> ir(irLenSamples);
  for (size_t i = 0; i < irLenSamples; ++i)
    ir[i] = std::exp(-3.0f * (float)i / (float)irLenSamples);

  // Run both engines and collect full output
  auto level0IrFFT = buildLevel0IrFFT(ir, blockSize);
  ConvolutionLevel level0(0, blockSize, 0, 2);
  TailEngine tail;
  tail.loadIR(ir.data(), ir.size(), blockSize);

  ConvolutionEngine stEngine;
  stEngine.loadIR(ir.data(), ir.size());

  std::vector<float> inBuf(blockSize, 0.0f);
  std::vector<float> mtOut(blockSize, 0.0f);
  std::vector<float> stOut(blockSize, 0.0f);

  float maxDiff = 0.0f;
  size_t maxDiffBlock = 0;
  bool firstBigDiffReported = false;

  for (size_t block = 0; block < numBlocks; ++block) {
    std::fill(inBuf.begin(), inBuf.end(), 0.0f);
    if (block == 0)
      inBuf[0] = 1.0f;

    level0.processBlock(inBuf.data(), level0IrFFT);
    tail.processBlock(inBuf.data());

    const float* level0Out = level0.getResult();
    const float* tl = tail.getResult();
    for (size_t i = 0; i < blockSize; ++i)
      mtOut[i] = level0Out[i] + tl[i];

    stEngine.process(inBuf.data(), stOut.data(), blockSize);

    // Check per-block max difference
    float blockMax = 0.0f;
    for (size_t i = 0; i < blockSize; ++i)
      blockMax = std::fmax(blockMax, std::fabs(mtOut[i] - stOut[i]));

    if (blockMax > maxDiff) {
      maxDiff = blockMax;
      maxDiffBlock = block;
    }

    // Report first block with large divergence
    if (blockMax > 1e-3f && !firstBigDiffReported) {
      float timeSec = (float)(block * blockSize) / sampleRate;
      printf("  First significant divergence at block %zu (%.2fs): "
             "max diff = %e\n",
             block,
             timeSec,
             blockMax);
      // Show a few samples
      for (size_t i = 0; i < std::min((size_t)8, blockSize); ++i) {
        if (std::fabs(mtOut[i] - stOut[i]) > 1e-4f)
          printf("    sample[%zu]: MT=%e  ST=%e  diff=%e\n",
                 i,
                 mtOut[i],
                 stOut[i],
                 mtOut[i] - stOut[i]);
      }
      firstBigDiffReported = true;
    }
  }

  printf("  Overall max diff: %e at block %zu (%.2fs)\n",
         maxDiff,
         maxDiffBlock,
         (float)(maxDiffBlock * blockSize) / sampleRate);

  check(maxDiff < 1e-2f, "MT vs ST max diff < 0.01 for 30s IR");
}

int
main()
{
  printf("=== Tail Dropout Diagnostic Tests ===\n");

  testTailDropout();
  testMTvsSTLongIR();

  printf("\n%d passed, %d failed\n", sTestsPassed, sTestsFailed);
  return sTestsFailed > 0 ? 1 : 0;
}
