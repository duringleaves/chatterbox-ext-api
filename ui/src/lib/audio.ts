const writeString = (view: DataView, offset: number, str: string) => {
  for (let i = 0; i < str.length; i += 1) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
};

const floatTo16BitPCM = (output: DataView, offset: number, input: Float32Array) => {
  for (let i = 0; i < input.length; i += 1, offset += 2) {
    const s = Math.max(-1, Math.min(1, input[i]));
    output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
};

const interleave = (buffers: Float32Array[]): Float32Array => {
  if (buffers.length === 1) {
    return buffers[0];
  }

  const length = buffers[0].length;
  const interleaved = new Float32Array(length * buffers.length);

  for (let channel = 0; channel < buffers.length; channel += 1) {
    const input = buffers[channel];
    for (let i = 0; i < length; i += 1) {
      interleaved[i * buffers.length + channel] = input[i];
    }
  }

  return interleaved;
};

const audioBufferToWav = (audioBuffer: AudioBuffer): ArrayBuffer => {
  const numChannels = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const channelData: Float32Array[] = [];

  for (let channel = 0; channel < numChannels; channel += 1) {
    channelData.push(audioBuffer.getChannelData(channel));
  }

  const interleaved = interleave(channelData);
  const buffer = new ArrayBuffer(44 + interleaved.length * 2);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + interleaved.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true); // Subchunk1Size for PCM
  view.setUint16(20, 1, true); // Audio format (PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * 2, true);
  view.setUint16(32, numChannels * 2, true);
  view.setUint16(34, 16, true); // Bits per sample
  writeString(view, 36, "data");
  view.setUint32(40, interleaved.length * 2, true);

  floatTo16BitPCM(view, 44, interleaved);

  return buffer;
};

export const convertBlobToWav = async (blob: Blob): Promise<Blob> => {
  if (typeof window === "undefined") {
    throw new Error("Audio conversion is not supported on the server");
  }

  const arrayBuffer = await blob.arrayBuffer();
  const AudioContextClass = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;

  if (!AudioContextClass) {
    throw new Error("AudioContext is not supported in this browser");
  }

  const audioContext = new AudioContextClass();

  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const wavBuffer = audioBufferToWav(audioBuffer);
    return new Blob([wavBuffer], { type: "audio/wav" });
  } finally {
    if (typeof audioContext.close === "function") {
      await audioContext.close();
    }
  }
};
