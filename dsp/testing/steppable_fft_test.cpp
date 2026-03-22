#include "../fft.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static const float EPSILON = 1e-3f;

static float
maxError(const float* a, const float* b, size_t count)
{
  float maxErr = 0.0f;
  for (size_t i = 0; i < count; i++)
    maxErr = std::max(maxErr, std::abs(a[i] - b[i]));
  return maxErr;
}

// Generate a test signal: sum of a few sinusoids
static void
generateTestSignal(float* buf, size_t size)
{
  for (size_t i = 0; i < size; i++) {
    float t = (float)i / (float)size;
    buf[i] = std::sin(2.0f * 3.14159265f * 3.0f * t) +
             0.5f * std::sin(2.0f * 3.14159265f * 17.0f * t) +
             0.3f * std::cos(2.0f * 3.14159265f * 53.0f * t);
  }
}

// Test: steppable FFT with all stages at once matches existing fft()
static bool
testFFTAllAtOnce(size_t size)
{
  std::vector<float> input(size);
  generateTestSignal(input.data(), size);

  // Reference
  std::vector<float> refOut(size * 2);
  fft(input.data(), refOut.data(), size);

  // Steppable — all stages at once
  std::vector<float> stepOut(size * 2);
  FFTState state;
  state.data = stepOut.data();
  state.size = size;
  fft_init(state, input.data());
  bool done = fft_steps(state, state.totalStages);
  assert(done);

  float err = maxError(refOut.data(), stepOut.data(), size * 2);
  printf("  FFT all-at-once size=%zu: maxErr=%.6e %s\n", size, err,
         err < EPSILON ? "PASS" : "FAIL");
  return err < EPSILON;
}

// Test: steppable FFT with N stages at a time matches existing fft()
static bool
testFFTStepped(size_t size, int stagesPerStep)
{
  std::vector<float> input(size);
  generateTestSignal(input.data(), size);

  // Reference
  std::vector<float> refOut(size * 2);
  fft(input.data(), refOut.data(), size);

  // Steppable
  std::vector<float> stepOut(size * 2);
  FFTState state;
  state.data = stepOut.data();
  state.size = size;
  fft_init(state, input.data());

  bool done = false;
  while (!done)
    done = fft_steps(state, stagesPerStep);

  float err = maxError(refOut.data(), stepOut.data(), size * 2);
  printf("  FFT %d-stage steps size=%zu: maxErr=%.6e %s\n", stagesPerStep,
         size, err, err < EPSILON ? "PASS" : "FAIL");
  return err < EPSILON;
}

// Test: steppable IFFT with all stages at once matches existing ifft()
static bool
testIFFTAllAtOnce(size_t size)
{
  // Create a spectrum by FFT-ing a test signal
  std::vector<float> input(size);
  generateTestSignal(input.data(), size);
  std::vector<float> spectrum(size * 2);
  fft(input.data(), spectrum.data(), size);

  // Reference IFFT
  std::vector<float> refOut(size * 2);
  ifft(spectrum.data(), refOut.data(), size);

  // Steppable IFFT — all stages at once
  std::vector<float> stepOut(size * 2);
  FFTState state;
  state.data = stepOut.data();
  state.size = size;
  ifft_init(state, spectrum.data());
  bool done = fft_steps(state, state.totalStages);
  assert(done);

  // Apply normalization for comparison
  float norm = 1.0f / (float)size;
  for (size_t i = 0; i < size * 2; i++)
    stepOut[i] *= norm;

  float err = maxError(refOut.data(), stepOut.data(), size * 2);
  printf("  IFFT all-at-once size=%zu: maxErr=%.6e %s\n", size, err,
         err < EPSILON ? "PASS" : "FAIL");
  return err < EPSILON;
}

// Test: steppable IFFT with N stages at a time matches existing ifft()
static bool
testIFFTStepped(size_t size, int stagesPerStep)
{
  std::vector<float> input(size);
  generateTestSignal(input.data(), size);
  std::vector<float> spectrum(size * 2);
  fft(input.data(), spectrum.data(), size);

  // Reference IFFT
  std::vector<float> refOut(size * 2);
  ifft(spectrum.data(), refOut.data(), size);

  // Steppable IFFT
  std::vector<float> stepOut(size * 2);
  FFTState state;
  state.data = stepOut.data();
  state.size = size;
  ifft_init(state, spectrum.data());

  bool done = false;
  while (!done)
    done = fft_steps(state, stagesPerStep);

  float norm = 1.0f / (float)size;
  for (size_t i = 0; i < size * 2; i++)
    stepOut[i] *= norm;

  float err = maxError(refOut.data(), stepOut.data(), size * 2);
  printf("  IFFT %d-stage steps size=%zu: maxErr=%.6e %s\n", stagesPerStep,
         size, err, err < EPSILON ? "PASS" : "FAIL");
  return err < EPSILON;
}

// Test: round-trip steppable FFT -> steppable IFFT matches original input
static bool
testRoundTrip(size_t size)
{
  std::vector<float> input(size);
  generateTestSignal(input.data(), size);

  // Forward FFT (steppable, 1 stage at a time)
  std::vector<float> spectrum(size * 2);
  FFTState fwdState;
  fwdState.data = spectrum.data();
  fwdState.size = size;
  fft_init(fwdState, input.data());
  while (!fft_steps(fwdState, 1))
    ;

  // Inverse FFT (steppable, 2 stages at a time)
  std::vector<float> ifftBuf(size * 2);
  FFTState invState;
  invState.data = ifftBuf.data();
  invState.size = size;
  ifft_init(invState, spectrum.data());
  while (!fft_steps(invState, 2))
    ;

  // Extract real output
  std::vector<float> output(size);
  ifft_extract_real(invState, output.data());

  float err = maxError(input.data(), output.data(), size);
  printf("  Round-trip size=%zu: maxErr=%.6e %s\n", size, err,
         err < EPSILON ? "PASS" : "FAIL");
  return err < EPSILON;
}

int
main()
{
  size_t sizes[] = { 256, 512, 2048, 16384, 65536 };
  int numSizes = sizeof(sizes) / sizeof(sizes[0]);
  int passed = 0, total = 0;

  for (int i = 0; i < numSizes; i++) {
    size_t sz = sizes[i];
    printf("--- Size %zu ---\n", sz);

    // FFT tests
    if (testFFTAllAtOnce(sz)) passed++;
    total++;
    if (testFFTStepped(sz, 1)) passed++;
    total++;
    if (testFFTStepped(sz, 2)) passed++;
    total++;

    // IFFT tests
    if (testIFFTAllAtOnce(sz)) passed++;
    total++;
    if (testIFFTStepped(sz, 1)) passed++;
    total++;
    if (testIFFTStepped(sz, 2)) passed++;
    total++;

    // Round-trip
    if (testRoundTrip(sz)) passed++;
    total++;
  }

  printf("\n=== %d / %d tests passed ===\n", passed, total);
  return passed == total ? 0 : 1;
}
