import { useRef } from "react";
import "./App.css";

const BLOCK = 128;
const MAX_RING_BLOCKS = 64;
// 3 ring buffers: dry, wetL, wetR (each MAX_RING_BLOCKS × BLOCK floats)
const SAB_SIZE = 8 + MAX_RING_BLOCKS * BLOCK * 3 * 4;

function App() {
  const audioContextRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);

  const initAudio = async () => {
    if (audioContextRef.current) return;

    const ctx = new AudioContext();

    const [audioEngineResponse, tailEngineResponse] = await Promise.all([
      fetch("/audio-engine.js"),
      fetch("/tail-engine.js"),
    ]);
    const audioEngineCode = await audioEngineResponse.text();
    const tailEngineCode = await tailEngineResponse.text();

    const sab = new SharedArrayBuffer(SAB_SIZE);

    const tailWorker = new Worker("/tail-worker.js");

    await new Promise<void>((resolve) => {
      tailWorker.onmessage = (e) => {
        if (e.data.type === "ready") resolve();
      };
      tailWorker.postMessage({ type: "init", scriptCode: tailEngineCode });
    });

    tailWorker.postMessage({ type: "sharedBuffer", sab });

    await ctx.audioWorklet.addModule("/dsp-processor.js");

    const node = new AudioWorkletNode(ctx, "dsp-processor", {
      outputChannelCount: [2],
    });
    node.connect(ctx.destination);

    await new Promise<void>((resolve) => {
      node.port.onmessage = (e) => {
        if (e.data.type === "ready") resolve();
        if (e.data.type === "error") {
          console.error(
            `Worklet error [${e.data.context}]:`,
            e.data.message,
            e.data.stack,
          );
        }
      };
      node.port.postMessage({ type: "init", scriptCode: audioEngineCode });
    });

    node.port.onmessage = (e) => {
      if (e.data.type === "error") {
        console.error(
          `Worklet error [${e.data.context}]:`,
          e.data.message,
          e.data.stack,
        );
      }
    };

    node.port.postMessage({ type: "sharedBuffer", sab });

    const irFile = await fetch("/ir30s.wav");
    const arrayBuffer = await irFile.arrayBuffer();
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer);

    const numChannels = audioBuffer.numberOfChannels;
    const length = audioBuffer.length;
    const irSamples = new Float32Array(length * numChannels);

    for (let ch = 0; ch < numChannels; ch++) {
      const channelData = audioBuffer.getChannelData(ch);
      for (let i = 0; i < length; i++) {
        irSamples[i * numChannels + ch] = channelData[i];
      }
    }

    const irMessage = {
      type: "loadIR",
      irSamples,
      irLength: length,
      numChannels,
    };

    tailWorker.postMessage(irMessage);
    await new Promise<void>((resolve) => {
      tailWorker.onmessage = (e) => {
        if (e.data.type === "irLoaded") {
          console.log("[worker] All IR levels loaded");
          resolve();
        }
      };
    });
    node.port.postMessage(irMessage);

    const kickFile = await fetch("/Arnold.wav");
    const kickArrayBuffer = await kickFile.arrayBuffer();
    const kickAudioBuffer = await ctx.decodeAudioData(kickArrayBuffer);
    const samples = kickAudioBuffer.getChannelData(0);
    node.port.postMessage({ type: "loadSample", samples });

    audioContextRef.current = ctx;
    workletNodeRef.current = node;
  };

  const handleCue = async () => {
    if (!audioContextRef.current) {
      await initAudio();
    }

    if (audioContextRef.current?.state === "suspended") {
      await audioContextRef.current.resume();
    }

    workletNodeRef.current?.port.postMessage({ type: "play" });
  };

  return (
    <div>
      <h1>Multithreaded Convolution Reverb</h1>
      <button onPointerDown={handleCue}>Cue</button>
    </div>
  );
}

export default App;
