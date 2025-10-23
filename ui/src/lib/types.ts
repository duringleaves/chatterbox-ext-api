export interface LineGenerationRequest {
  line_id: string;
  text: string;
  section?: string;
  reference_voice: string;
  reference_style: string;
  reference_audio: string;
  options: TTSOptions;
  tag?: string | null;
  sound_words_field?: string | null;
  clone_voice?: string | null;
  clone_audio?: string | null;
  clone_pitch?: number;
  job_id?: string | null;
  queue_position?: number | null;
}

export interface TTSOptions {
  exaggeration: number;
  temperature: number;
  seed: number;
  cfg_weight: number;
  use_pyrnnoise: boolean;
  use_auto_editor: boolean;
  auto_editor_threshold: number;
  auto_editor_margin: number;
  export_formats: string[];
  enable_batching: boolean;
  smart_batch_short_sentences: boolean;
  to_lowercase: boolean;
  normalize_spacing: boolean;
  fix_dot_letters: boolean;
  remove_reference_numbers: boolean;
  keep_original_wav: boolean;
  disable_watermark: boolean;
  num_generations: number;
  normalize_audio: boolean;
  normalize_method: string;
  normalize_level: number;
  normalize_true_peak: number;
  normalize_lra: number;
  num_candidates: number;
  max_attempts: number;
  bypass_whisper: boolean;
  whisper_model: string;
  enable_parallel: boolean;
  num_parallel_workers: number;
  use_longest_transcript_on_fail: boolean;
  sound_words_field?: string | null;
  sound_words?: { pattern: string; replacement?: string }[] | null;
  use_faster_whisper: boolean;
  generate_separate_audio_files: boolean;
}

export interface StationFormatDescriptor {
  id: string;
  filename: string;
}

export interface SampleStationDescriptor {
  id: string;
  filename: string;
}

export interface SampleScriptDescriptor {
  id: string;
  filename: string;
}

export interface ReferenceVoiceStyle {
  name: string;
  audio_files: string[];
  default_settings: Record<string, unknown> | null;
  tag_settings: Record<string, Record<string, unknown>>;
}

export interface ReferenceVoice {
  name: string;
  styles: ReferenceVoiceStyle[];
}

export interface CloneVoice {
  name: string;
  files: string[];
}

export interface AnalyzeResponse {
  processed_lines: string[];
}

export type LineStatus = "pending" | "processing" | "completed" | "failed" | "cancelled";
export type JobState = "pending" | "running" | "completed" | "cancelled" | "failed";

export interface FileResult {
  path: string;
  url?: string;
  size_bytes: number;
  duration_seconds?: number;
  base64?: string;
}

export interface LineGenerationResponse {
  line_id: string;
  raw_outputs: FileResult[];
  final_outputs: FileResult[];
  metadata: Record<string, string>;
  zip_file?: FileResult | null;
}

export interface BatchLineStatus {
  line_id: string;
  status: LineStatus;
  error?: string | null;
  raw_outputs?: FileResult[] | null;
  final_outputs?: FileResult[] | null;
}

export interface BatchJobStatus {
  job_id: string;
  state: JobState;
  progress: number;
  total_lines: number;
  completed_lines: number;
  failed_lines: number;
  lines: BatchLineStatus[];
  zip_file?: FileResult | null;
  message?: string | null;
}
