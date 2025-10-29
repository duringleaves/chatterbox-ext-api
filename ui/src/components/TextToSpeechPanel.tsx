import { ChangeEvent, useEffect, useMemo, useState } from "react";
import {
  Alert,
  Badge,
  Button,
  Card,
  Collapse,
  Divider,
  Group,
  Loader,
  Select,
  Slider,
  Stack,
  Text,
  Textarea,
  Title
} from "@mantine/core";
import { IconDownload, IconPlayerPlay } from "@tabler/icons-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import {
  CloneVoice,
  FileResult,
  LineGenerationResponse,
  ReferenceVoice,
  ReferenceVoiceStyle,
  TTSOptions
} from "@/lib/types";
import { useApiKey } from "@/hooks/useApiKey";

const defaultsToOptions = (defaults: Record<string, any>): TTSOptions => ({
  exaggeration: defaults.exaggeration_slider ?? 0.5,
  temperature: defaults.temp_slider ?? 0.75,
  seed: defaults.seed_input ?? 0,
  cfg_weight: defaults.cfg_weight_slider ?? 1.0,
  use_pyrnnoise: defaults.use_pyrnnoise_checkbox ?? false,
  use_auto_editor: defaults.use_auto_editor_checkbox ?? false,
  auto_editor_threshold: defaults.threshold_slider ?? 0.06,
  auto_editor_margin: defaults.margin_slider ?? 0.2,
  export_formats: defaults.export_format_checkboxes ?? ["wav"],
  enable_batching: defaults.enable_batching_checkbox ?? false,
  smart_batch_short_sentences: defaults.smart_batch_short_sentences_checkbox ?? true,
  to_lowercase: defaults.to_lowercase_checkbox ?? true,
  normalize_spacing: defaults.normalize_spacing_checkbox ?? true,
  fix_dot_letters: defaults.fix_dot_letters_checkbox ?? true,
  remove_reference_numbers: defaults.remove_reference_numbers_checkbox ?? true,
  keep_original_wav: defaults.keep_original_checkbox ?? false,
  disable_watermark: defaults.disable_watermark_checkbox ?? true,
  num_generations: defaults.num_generations_input ?? 1,
  normalize_audio: defaults.normalize_audio_checkbox ?? false,
  normalize_method: defaults.normalize_method_dropdown ?? "ebu",
  normalize_level: defaults.normalize_level_slider ?? -24,
  normalize_true_peak: defaults.normalize_tp_slider ?? -2,
  normalize_lra: defaults.normalize_lra_slider ?? 7,
  num_candidates: defaults.num_candidates_slider ?? 3,
  max_attempts: defaults.max_attempts_slider ?? 3,
  bypass_whisper: defaults.bypass_whisper_checkbox ?? false,
  whisper_model: defaults.whisper_model_dropdown ?? "medium (~5–8 GB OpenAI / ~2.5–4.5 GB faster-whisper)",
  enable_parallel: defaults.enable_parallel_checkbox ?? true,
  num_parallel_workers: defaults.num_parallel_workers_slider ?? 4,
  use_longest_transcript_on_fail: defaults.use_longest_transcript_on_fail_checkbox ?? true,
  sound_words_field: defaults.sound_words_field ?? "",
  sound_words: [],
  use_faster_whisper: defaults.use_faster_whisper_checkbox ?? true,
  generate_separate_audio_files: defaults.separate_files_checkbox ?? false,
  force_reference_defaults: true
});

const AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".m4a"];
const isAudioFile = (file: string) => AUDIO_EXTENSIONS.some((ext) => file.toLowerCase().endsWith(ext));

const applyStyleOverrides = (
  options: TTSOptions,
  style: ReferenceVoiceStyle | undefined,
  tag?: string | null
): TTSOptions => {
  const merged = { ...options };
  if (style?.default_settings) {
    const defaults = style.default_settings;
    if (typeof defaults.temperature === "number") merged.temperature = defaults.temperature;
    if (typeof defaults.exaggeration === "number") merged.exaggeration = defaults.exaggeration;
    if (typeof defaults.cfg_weight === "number") merged.cfg_weight = defaults.cfg_weight;
  }
  if (tag) {
    const tagSettings = style?.tag_settings?.[tag.toLowerCase()];
    if (tagSettings) {
      if (typeof tagSettings.temperature === "number") merged.temperature = tagSettings.temperature;
      if (typeof tagSettings.exaggeration === "number") merged.exaggeration = tagSettings.exaggeration;
      if (typeof tagSettings.cfg_weight === "number") merged.cfg_weight = tagSettings.cfg_weight;
    }
  }
  return merged;
};

const getReferenceUrl = (voice: string | null, style: string | null, file: string | null, apiKey?: string | null) => {
  if (!voice || !style || !file) return undefined;
  const encoded = encodeURIComponent(file);
  const base = `/voices/reference/${encodeURIComponent(voice)}/${encodeURIComponent(style)}/${encoded}`;
  if (!apiKey) return base;
  return `${base}?api_key=${encodeURIComponent(apiKey)}`;
};

const PlayButton = ({
  url,
  onPlay,
  disabled = false,
  label = "Preview"
}: {
  url?: string;
  onPlay?: (url: string) => void;
  disabled?: boolean;
  label?: string;
}) => (
  <Button
    size="xs"
    variant="light"
    leftSection={<IconPlayerPlay size={14} />}
    disabled={disabled || !url}
    onClick={() => url && !disabled && onPlay?.(url)}
  >
    {label}
  </Button>
);

export const TextToSpeechPanel = () => {
  const defaultsQuery = useQuery<Record<string, any>>({
    queryKey: ["tts-defaults"],
    queryFn: async () => (await api.get("/tts/defaults")).data
  });

  const referenceVoicesQuery = useQuery<ReferenceVoice[]>({
    queryKey: ["reference-voices"],
    queryFn: async () => (await api.get<ReferenceVoice[]>("/voices/reference")).data
  });

  const cloneVoicesQuery = useQuery<CloneVoice[]>({
    queryKey: ["clone-voices"],
    queryFn: async () => (await api.get<CloneVoice[]>("/voices/clone")).data
  });

  const { apiKey } = useApiKey();
  const [inputText, setInputText] = useState<string>("");
  const [selectedVoice, setSelectedVoice] = useState<string | null>(null);
  const [selectedStyle, setSelectedStyle] = useState<string | null>(null);
  const [selectedReferenceAudio, setSelectedReferenceAudio] = useState<string | null>(null);
  const [selectedTag, setSelectedTag] = useState<string>("");
  const [cloneVoice, setCloneVoice] = useState<string | null>(null);
  const [cloneSample, setCloneSample] = useState<string | null>(null);
  const [takesPerLine, setTakesPerLine] = useState<number>(1);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [latestOutputs, setLatestOutputs] = useState<FileResult[]>([]);
  const [latestRawOutputs, setLatestRawOutputs] = useState<FileResult[]>([]);
  const [cloneOptionsOpen, setCloneOptionsOpen] = useState<boolean>(false);

  const defaults = defaultsQuery.data;
  const referenceVoices = referenceVoicesQuery.data ?? [];
  const cloneVoices = cloneVoicesQuery.data ?? [];

  const baseOptions = useMemo(() => (defaults ? defaultsToOptions(defaults) : null), [defaults]);

  const activeVoice = referenceVoices.find((voice) => voice.name === selectedVoice);
  const activeStyle = activeVoice?.styles.find((style) => style.name === selectedStyle);
  const tagOptions = activeStyle
    ? [
        { value: "", label: "Default" },
        ...Object.keys(activeStyle.tag_settings).map((tag) => ({ value: tag, label: tag }))
      ]
    : [{ value: "", label: "Default" }];

  const referenceAudioOptions =
    activeStyle?.audio_files.filter(isAudioFile).map((file) => ({
      value: file,
      label: file
    })) ?? [];

  const cloneVoiceOptions = cloneVoices.map((voice) => ({ value: voice.name, label: voice.name }));
  const cloneSampleOptions =
    cloneVoices.find((voice) => voice.name === cloneVoice)?.files.map((file) => ({
      value: file,
      label: file
    })) ?? [];

  useEffect(() => {
    if (!selectedVoice && referenceVoices.length > 0) {
      setSelectedVoice(referenceVoices[0].name);
    }
  }, [referenceVoices, selectedVoice]);

  useEffect(() => {
    if (!selectedVoice) return;
    const voice = referenceVoices.find((v) => v.name === selectedVoice);
    if (!voice) return;
    const preferred =
      voice.styles.find((style) => style.name.toLowerCase() === "normal") ?? voice.styles[0] ?? null;
    setSelectedStyle(preferred?.name ?? null);
    setSelectedTag("");
  }, [referenceVoices, selectedVoice]);

  useEffect(() => {
    if (!activeStyle) {
      setSelectedReferenceAudio(null);
      return;
    }
    const firstAudio = activeStyle.audio_files.find(isAudioFile) ?? null;
    setSelectedReferenceAudio((prev) => {
      if (prev && activeStyle.audio_files.includes(prev) && isAudioFile(prev)) {
        return prev;
      }
      return firstAudio;
    });
  }, [activeStyle]);

  useEffect(() => {
    if (cloneVoice) return;
    setCloneSample(null);
  }, [cloneVoice]);

  useEffect(() => {
    if (!tagOptions.some((option) => option.value === selectedTag)) {
      setSelectedTag(tagOptions[0]?.value ?? "");
    }
  }, [tagOptions, selectedTag]);

  const analyzeMutation = useMutation({
    mutationFn: async () => {
      const text = inputText.trim();
      if (!text) throw new Error("Provide text before analyzing");
      const res = await api.post<{ processed_lines: string[] }>("/scripts/analyze", { lines: [text] });
      return res.data.processed_lines[0] ?? text;
    },
    onMutate: () => {
      setAnalyzeError(null);
    },
    onSuccess: (processed) => {
      setInputText(processed);
    },
    onError: (error: unknown) => {
      if (error && typeof error === "object" && "response" in error && (error as any).response) {
        const axiosError = error as { response?: { data?: any } };
        const detail = axiosError.response?.data?.detail;
        if (typeof detail === "string") {
          setAnalyzeError(detail);
          return;
        }
      }
      if (error instanceof Error) {
        setAnalyzeError(error.message);
        return;
      }
      setAnalyzeError("Failed to analyze text. Please try again.");
    }
  });

  const generationMutation = useMutation({
    mutationFn: async () => {
      const usedCloneVoice = cloneVoice;
      if (!baseOptions) throw new Error("Defaults not loaded");
      const trimmed = inputText.trim();
      if (!trimmed) throw new Error("Provide text to generate");
      if (!selectedVoice) throw new Error("Select a reference voice");
      if (!selectedStyle) throw new Error("Select a read style");
      if (!selectedReferenceAudio) throw new Error("Select a reference audio file");
      const voice = referenceVoices.find((v) => v.name === selectedVoice);
      const style = voice?.styles.find((s) => s.name === selectedStyle);
      const options = applyStyleOverrides({ ...baseOptions }, style, selectedTag || null);
      options.sound_words_field = baseOptions.sound_words_field ?? "";
      options.export_formats = ["wav"];
      const repeatedText = takesPerLine > 1 ? Array.from({ length: takesPerLine }, () => trimmed).join("\n") : trimmed;
      const payload = {
        line_id: `tts-${Date.now()}`,
        text: repeatedText,
        section: "Ad-hoc",
        reference_voice: selectedVoice,
        reference_style: selectedStyle,
        reference_audio: selectedReferenceAudio,
        tag: selectedTag || undefined,
        clone_voice: cloneVoice ?? undefined,
        clone_audio: cloneSample ?? undefined,
        clone_pitch: 0,
        options
      };
      const res = await api.post<LineGenerationResponse>("/lines/generate", payload);
      return { response: res.data, usedCloneVoice };
    },
    onMutate: () => {
      setGenerationError(null);
      setLatestOutputs([]);
      setLatestRawOutputs([]);
    },
    onSuccess: ({ response, usedCloneVoice }) => {
      const onlyWav = (files?: FileResult[] | null) =>
        (files ?? []).filter((file) => file.path.toLowerCase().endsWith(".wav"));
      setLatestOutputs(usedCloneVoice ? onlyWav(response.final_outputs) : []);
      setLatestRawOutputs(onlyWav(response.raw_outputs));
    },
    onError: (error: unknown) => {
      if (error && typeof error === "object" && "response" in error && (error as any).response) {
        const axiosError = error as { response?: { data?: any } };
        const detail = axiosError.response?.data?.detail;
        if (typeof detail === "string") {
          setGenerationError(detail);
          setLatestOutputs([]);
          setLatestRawOutputs([]);
          return;
        }
      }
      const message = error instanceof Error ? error.message : "Failed to generate audio";
      setGenerationError(message);
      setLatestOutputs([]);
      setLatestRawOutputs([]);
    }
  });

  const handleTextChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    setInputText(event.currentTarget.value);
  };

  const handlePreview = (url?: string) => {
    if (!url) return;
    const withTs = url.includes("?") ? `${url}&ts=${Date.now()}` : `${url}?ts=${Date.now()}`;
    setPreviewUrl(withTs);
  };

  const referenceUrl = getReferenceUrl(selectedVoice, selectedStyle, selectedReferenceAudio, apiKey);

  const renderLoader = defaultsQuery.isLoading || referenceVoicesQuery.isLoading;

  return (
    <Stack gap="lg">
      <Card withBorder padding="lg" radius="md" shadow="sm">
        <Stack gap="sm">
          <Title order={4}>Input Phrase</Title>
          <Textarea
            placeholder="Enter text to render"
            autosize
            minRows={4}
            maxRows={12}
            value={inputText}
            onChange={handleTextChange}
          />
          <Group gap="sm">
            <Button
              variant="light"
              color="violet"
              onClick={() => analyzeMutation.mutate()}
              loading={analyzeMutation.isPending}
              disabled={!inputText.trim()}
            >
              Analyze with ChatGPT
            </Button>
            {analyzeError && (
              <Text size="sm" c="red.4">
                {analyzeError}
              </Text>
            )}
          </Group>
        </Stack>
      </Card>

      <Card withBorder padding="lg" radius="md" shadow="sm">
        <Stack gap="md">
          <Title order={4}>Voice Settings</Title>
          {renderLoader ? (
            <Loader />
          ) : (
            <>
              <Group grow>
                <Select
                  label="Reference voice"
                  data={referenceVoices.map((voice) => ({ value: voice.name, label: voice.name }))}
                  value={selectedVoice}
                  onChange={(voice) => {
                    setSelectedVoice(voice);
                    setSelectedStyle(null);
                    setSelectedReferenceAudio(null);
                    setSelectedTag("");
                  }}
                  placeholder="Select voice"
                />
                <Select
                  label="Read style"
                  data={activeVoice?.styles.map((style) => ({ value: style.name, label: style.name })) ?? []}
                  value={selectedStyle}
                  onChange={(style) => {
                    setSelectedStyle(style);
                    setSelectedReferenceAudio(null);
                    setSelectedTag("");
                  }}
                  disabled={!selectedVoice}
                  placeholder="Select style"
                />
              </Group>
              <Group grow>
                <Select
                  label="Tag"
                  placeholder="Select tag"
                  data={tagOptions}
                  value={selectedTag}
                  onChange={(tag) => setSelectedTag(tag ?? "")}
                />
                <Select
                  label="Reference audio"
                  placeholder={referenceAudioOptions.length ? "Select reference" : "No files available"}
                  data={referenceAudioOptions}
                  value={selectedReferenceAudio}
                  onChange={setSelectedReferenceAudio}
                  disabled={referenceAudioOptions.length === 0}
                  searchable
                />
                <PlayButton
                  url={referenceUrl}
                  onPlay={handlePreview}
                  disabled={!selectedReferenceAudio}
                  label="Preview reference"
                />
              </Group>
            </>
          )}
        </Stack>
      </Card>

      <Card withBorder padding="lg" radius="md" shadow="sm">
        <Stack gap="sm">
          <Group justify="space-between" align="center">
            <Title order={4}>Clone Options</Title>
            <Button
              variant="subtle"
              size="xs"
              color="violet"
              onClick={() => setCloneOptionsOpen((prev) => !prev)}
            >
              {cloneOptionsOpen ? "Hide" : "Show"}
            </Button>
          </Group>
          <Collapse in={cloneOptionsOpen}>
            <Stack gap="md">
              <Group grow>
                <Select
                  label="Clone voice"
                  placeholder="None"
                  data={[{ value: "", label: "None" }, ...cloneVoiceOptions]}
                  value={cloneVoice ?? ""}
                  onChange={(value) => setCloneVoice(value || null)}
                />
                <Select
                  label="Clone sample"
                  placeholder="Auto"
                  data={cloneSampleOptions}
                  value={cloneSample}
                  onChange={setCloneSample}
                  disabled={!cloneVoice}
                />
              </Group>
            </Stack>
          </Collapse>
        </Stack>
      </Card>

      <Card withBorder padding="lg" radius="md" shadow="sm">
        <Stack gap="md">
          <Title order={4}>Render</Title>
          <Stack gap="sm">
            <Stack gap={4}>
              <Text fw={500}>Takes per render ({takesPerLine})</Text>
              <Slider
                min={1}
                max={3}
                step={1}
                value={takesPerLine}
                onChange={setTakesPerLine}
                marks={[
                  { value: 1, label: "1" },
                  { value: 2, label: "2" },
                  { value: 3, label: "3" }
                ]}
              />
            </Stack>
            <Group>
              <Button
                color="violet"
                onClick={() => generationMutation.mutate()}
                loading={generationMutation.isPending}
                disabled={!inputText.trim() || !selectedVoice || !selectedStyle || !selectedReferenceAudio}
              >
                Generate Audio
              </Button>
              {generationError && (
                <Text size="sm" c="red.4">
                  {generationError}
                </Text>
              )}
            </Group>
          </Stack>

          {latestOutputs.length > 0 ? (
            <Stack gap="xs">
              <Divider label="Rendered Files" labelPosition="left" />
              <Stack gap="xs">
                {latestOutputs.map((file) => (
                  <Group key={file.path} justify="space-between" align="center">
                    <Stack gap={2}>
                      <Text size="sm">{file.path}</Text>
                      <Group gap="xs">
                        <Badge color="teal">Final</Badge>
                        {typeof file.duration_seconds === "number" && (
                          <Badge color="gray">{file.duration_seconds.toFixed(1)}s</Badge>
                        )}
                      </Group>
                    </Stack>
                    <Group gap="xs">
                      {isAudioFile(file.path) && file.url ? (
                        <PlayButton url={file.url} onPlay={handlePreview} />
                      ) : null}
                      <Button
                        component="a"
                        href={file.url ?? undefined}
                        variant="light"
                        color="teal"
                        leftSection={<IconDownload size={14} />}
                        disabled={!file.url}
                      >
                        Download
                      </Button>
                    </Group>
                  </Group>
                ))}
              </Stack>
            </Stack>
          ) : null}

          {latestRawOutputs.length > 0 ? (
            <Stack gap="xs">
              <Divider label="Raw Outputs" labelPosition="left" />
              <Stack gap="xs">
                {latestRawOutputs.map((file) => (
                  <Group key={file.path} justify="space-between" align="center">
                    <Text size="sm">{file.path}</Text>
                    <Group gap="xs">
                      {isAudioFile(file.path) && file.url ? (
                        <PlayButton url={file.url} onPlay={handlePreview} label="Preview raw" />
                      ) : null}
                      <Button
                        component="a"
                        href={file.url ?? undefined}
                        variant="light"
                        color="gray"
                        leftSection={<IconDownload size={14} />}
                        disabled={!file.url}
                      >
                        Download
                      </Button>
                    </Group>
                  </Group>
                ))}
              </Stack>
            </Stack>
          ) : null}
        </Stack>
      </Card>

      {previewUrl && (
        <audio
          key={previewUrl}
          src={previewUrl}
          autoPlay
          onEnded={() => setPreviewUrl(null)}
          style={{ display: "none" }}
        />
      )}

      {(defaultsQuery.isError || referenceVoicesQuery.isError || cloneVoicesQuery.isError) && (
        <Alert color="red" title="Failed to load configuration">
          Some reference data could not be loaded. Check the API connection and try again.
        </Alert>
      )}
    </Stack>
  );
};
