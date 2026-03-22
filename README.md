# Multithreaded Convolution Reverb

Real-time multithreaded convolution reverb running a 30-second impulse response in the browser.

## [Live Demo](https://multithreaded-convolution-demo.vercel.app/)

## What Was Done

I started out by building simplified pieces to gain a deeper understanding of exactly what is happening:

**Built by hand:**
- DFT and IDFT (`naive_convolution/dft.ts`, `idft.ts`)
- Naive linear and circular convolution (`naive_convolution/linear-convolve.ts`, `circular-convolve.ts`)
  - Naive because it does not perform convolution by block. performs on entire buffer
- Overlap-add convolution (`naive_convolution/overlap-add-convolver.ts`)
- Recursive FFT (`dsp/fft_recursive.cpp`)
- Single-threaded convolution reverb (`dsp/convolution_mine_st.cpp`)

**Then:** Read chapters 5 and 6 of 'Partitioned convolution algorithms for real-time auralization' by Frank Wefers.

**Then:** I used the info from this book (specifically the Gardner partitions) to design the architecture described briefly below and implement it with Claude.

## Architecture

Work is split across 3 threads:
- **Main thread** — UI
- **Audio thread** (AudioWorklet) — sampler and first block of convolution reverb (level 0)
- **Web Worker** — reverb tail worker that computes tail in Gardner partitions

Non-uniform partitions are implemented using Gardner partitions:
- 1, 1, 2, 2, 4, 4, ..., cap
  - early partitions are shorter so that the tail will be ready by the time it is needed
  - later partitions are longer because they have more time before they need to be ready
- For bigger partitions, the large FFTs are broken into smaller pieces so work is spread evenly
- Partitions are staggered to avoid thundering herd problem
- Audio thread and worker communicate via SharedArrayBuffer 
  - originally I tried to use post messages with transferables, but this was causing latency

## What's next

Probably doing a few laps reimplementing this by hand to fully understand the fine details of
- using a SharedArrayBuffer to pass information between threads
- partial FFT and state machine/pipeline to break large partitions into a manageable amount of work per block
- how to manage the indexing cleanly with different size partitions, breaking those into groups, and staggering

Going through DAFX book and practicing implementing each effect in the book by hand one at a time.

Kick With Reverb
- replace JUCE with from scratch C++ everywhere
- add automation
- add sidechaining
- add 303 with midi sequencer

## Build

```bash
emcmake cmake -B build
cmake --build build
cd frontend && npm install && npm run dev
```
