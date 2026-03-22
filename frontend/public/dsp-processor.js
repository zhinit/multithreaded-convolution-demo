const BLOCK = 128;
const MAX_RING_BLOCKS = 64;
const OUTPUT_GAIN = 0.3;

class DSPProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.engine = null;
    this.module = null;
    this.heapBufferLeft = null;
    this.heapBufferRight = null;

    this.dryWriteCount = null;
    this.wetWriteCount = null;
    this.dryRing = null;
    this.wetRingL = null;
    this.wetRingR = null;
    this.writeCount = 0;

    this.irLoaded = false;
    this.wetReadPos = 0;

    this.port.onmessage = (e) => this.handleMessage(e.data);
  }

  async handleMessage(data) {
    try {
      if (data.type === "init") {
        const fn = new Function(
          data.scriptCode + "; return createAudioEngine;",
        );
        const createAudioEngine = fn();
        const module = await createAudioEngine();
        this.engine = new module.Sampler();
        this.engine.prepare(sampleRate);
        this.module = module;
        this.port.postMessage({ type: "ready" });
      }
      if (data.type === "sharedBuffer") {
        const sab = data.sab;
        this.dryWriteCount = new Int32Array(sab, 0, 1);
        this.wetWriteCount = new Int32Array(sab, 4, 1);
        this.dryRing = new Float32Array(sab, 8, MAX_RING_BLOCKS * BLOCK);
        const wetLOffset = 8 + MAX_RING_BLOCKS * BLOCK * 4;
        const wetROffset = wetLOffset + MAX_RING_BLOCKS * BLOCK * 4;
        this.wetRingL = new Float32Array(
          sab,
          wetLOffset,
          MAX_RING_BLOCKS * BLOCK,
        );
        this.wetRingR = new Float32Array(
          sab,
          wetROffset,
          MAX_RING_BLOCKS * BLOCK,
        );
      }
      if (data.type === "loadSample") {
        const samplePtr = this.module._malloc(data.samples.length * 4);
        this.module.HEAPF32.set(data.samples, samplePtr / 4);
        this.engine?.loadSample(samplePtr, data.samples.length);
      }
      if (data.type === "play") {
        this.engine?.trigger();
      }
      if (data.type === "loadIR") {
        const irPtr = this.module._malloc(data.irSamples.length * 4);
        this.module.HEAPF32.set(data.irSamples, irPtr / 4);
        this.engine.prepareLevel0(irPtr, data.irLength, data.numChannels);
        this.module._free(irPtr);
        this.irLoaded = true;
      }
    } catch (e) {
      this.port.postMessage({
        type: "error",
        context: "handleMessage",
        message: e.message,
        stack: e.stack,
      });
    }
  }

  process(inputs, outputs, parameters) {
    if (!this.engine || !this.module) return true;

    try {
      const leftOutput = outputs[0][0];
      const rightOutput = outputs[0][1];
      const numSamples = leftOutput.length;

      if (!this.heapBufferLeft) {
        this.heapBufferLeft = this.module._malloc(numSamples * 4);
        this.heapBufferRight = this.module._malloc(numSamples * 4);
      }

      this.engine.process(
        this.heapBufferLeft,
        this.heapBufferRight,
        numSamples,
      );

      const dryLeft = new Float32Array(
        this.module.HEAPF32.buffer,
        this.heapBufferLeft,
        numSamples,
      );
      const dryRight = new Float32Array(
        this.module.HEAPF32.buffer,
        this.heapBufferRight,
        numSamples,
      );

      if (this.irLoaded && this.dryWriteCount) {
        // write dry block to shared ring for worker
        const dryPtr = this.engine.getDryBlock();
        const dryMono = new Float32Array(
          this.module.HEAPF32.buffer,
          dryPtr,
          numSamples,
        );
        const ringIdx = (this.writeCount % MAX_RING_BLOCKS) * BLOCK;
        this.dryRing.set(dryMono, ringIdx);
        this.writeCount++;
        Atomics.store(this.dryWriteCount, 0, this.writeCount);
        Atomics.notify(this.dryWriteCount, 0);

        // level 0 result (computed on audio thread)
        const l0lPtr = this.engine.getLevel0Left();
        const l0rPtr = this.engine.getLevel0Right();
        const l0l = new Float32Array(
          this.module.HEAPF32.buffer,
          l0lPtr,
          numSamples,
        );
        const l0r = new Float32Array(
          this.module.HEAPF32.buffer,
          l0rPtr,
          numSamples,
        );

        const wetWriteCount = Atomics.load(this.wetWriteCount, 0);

        if (this.wetReadPos < wetWriteCount) {
          const wetOffset = (this.wetReadPos % MAX_RING_BLOCKS) * BLOCK;
          this.wetReadPos++;
          for (let i = 0; i < numSamples; i++) {
            const wetL = l0l[i] + this.wetRingL[wetOffset + i];
            const wetR = l0r[i] + this.wetRingR[wetOffset + i];
            leftOutput[i] = OUTPUT_GAIN * (dryLeft[i] + wetL);
            rightOutput[i] = OUTPUT_GAIN * (dryRight[i] + wetR);
          }
        } else {
          // no tail data yet — level 0 + dry only
          for (let i = 0; i < numSamples; i++) {
            leftOutput[i] = OUTPUT_GAIN * (dryLeft[i] + l0l[i]);
            rightOutput[i] = OUTPUT_GAIN * (dryRight[i] + l0r[i]);
          }
        }
      } else {
        // no IR loaded — dry only
        for (let i = 0; i < numSamples; i++) {
          leftOutput[i] = OUTPUT_GAIN * dryLeft[i];
          rightOutput[i] = OUTPUT_GAIN * dryRight[i];
        }
      }
    } catch (e) {
      this.port.postMessage({
        type: "error",
        context: "process",
        message: e.message,
        stack: e.stack,
      });
    }

    return true;
  }
}

registerProcessor("dsp-processor", DSPProcessor);
