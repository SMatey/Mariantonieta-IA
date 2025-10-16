declare module 'wav-encoder' {
  export function wavEncode(
    audioData: Uint8Array,
    options: {
      sampleRate: number;
      channels: number;
      bitDepth: number;
    }
  ): Promise<Uint8Array>;
}