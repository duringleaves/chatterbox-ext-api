import { ChangeEvent, Fragment, useEffect, useMemo, useState } from "react";
import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Card,
  Divider,
  FileInput,
  Flex,
  Group,
  Loader,
  Select,
  Slider,
  Space,
  Stack,
  Table,
  Tabs,
  Text,
  Textarea,
  TextInput,
  Title
} from "@mantine/core";
import { useMutation, useQuery } from "@tanstack/react-query";
import { IconAlertCircle, IconChecks, IconDownload, IconPlayerPlay } from "@tabler/icons-react";
import { api } from "@/lib/api";
import {
  AnalyzeResponse,
  BatchCreateRequest,
  BatchJobStatus,
  CloneVoice,
  FileResult,
  LineGenerationResponse,
  LineStatus,
  ReferenceVoice,
  ReferenceVoiceStyle,
  SampleStationDescriptor,
  StationFormatDescriptor,
  TTSOptions
} from "@/lib/types";
import classes from "./VoKitPanel.module.css";
import { useApiKey } from "@/hooks/useApiKey";

interface ScriptLine {
  id: string;
  section: string;
  text: string;
  baseText: string;
  tag?: string | null;
  referenceKey?: string | null;
  referenceAudio?: string | null;
  soundWordsField?: string | null;
  status: LineStatus;
  rawOutputs?: FileResult[];
  finalOutputs?: FileResult[];
  error?: string | null;
}

const sanitizeSection = (section: string) => section.replace(/[<>]/g, "").trim();

const extractTag = (text: string): { tag?: string; clean: string } => {
  const tagMatch = text.match(/^\s*\[([^\]]+)\]/);
  if (!tagMatch) {
    return { clean: text.trim() };
  }
  const clean = text.replace(tagMatch[0], "").trim();
  return { tag: tagMatch[1].toLowerCase(), clean };
};

const parsePlainScript = (content: string): ScriptLine[] => {
  const lines: ScriptLine[] = [];
  let currentSection = "Misc";
  content.split(/\r?\n/).forEach((raw, index) => {
    const trimmed = raw.trim();
    if (!trimmed) return;
    if (/^<.+>$/.test(trimmed)) {
      currentSection = sanitizeSection(trimmed);
      return;
    }
    const { tag, clean } = extractTag(trimmed);
    lines.push({
      id: `line-${index}`,
      section: currentSection,
      text: clean,
      baseText: clean,
      tag,
      status: "pending"
    });
  });
  return lines;
};

const extractTemplateKeys = (template: Record<string, { text: string }[]>) => {
  const keys = new Set<string>();
  const regex = /%([^%]+)%/g;
  Object.values(template).forEach((entries) => {
    entries.forEach((entry) => {
      const text = entry.text;
      let match: RegExpExecArray | null;
      while ((match = regex.exec(text))) {
        keys.add(match[1]);
      }
    });
  });
  return Array.from(keys).sort();
};

const substituteTemplate = (
  template: Record<string, { text: string; reference?: string }[]>,
  values: Record<string, string>
): ScriptLine[] => {
  const result: ScriptLine[] = [];
  let index = 0;
  for (const [section, entries] of Object.entries(template)) {
    entries.forEach((entry) => {
      const { tag, clean } = extractTag(entry.text);
      const text = clean.replace(/%([^%]+)%/g, (_, key) => values[key] ?? `%${key}%`);
      result.push({
        id: `tmpl-${index++}`,
        section,
        text,
        baseText: text,
        tag,
        referenceKey: entry.reference,
        status: "pending"
      });
    });
  }
  return result;
};

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
  generate_separate_audio_files: defaults.separate_files_checkbox ?? false
});

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

const VARIANT_SUFFIX_MAP: Record<string, string> = {
  fast: "fast",
  slow: "slow",
  intense: "intense"
};

const pickReferenceAudio = (
  voice: ReferenceVoice | undefined,
  styleName: string | null,
  key: string | null | undefined,
  tag: string | null | undefined,
  current: string | null | undefined,
  seed: string
): string | null => {
  const style = voice?.styles.find((s) => s.name === styleName);
  if (!style || !style.audio_files.length) {
    return current ?? null;
  }

  const audioFiles = style.audio_files;
  const normalizedKey = key?.toLowerCase().trim() ?? null;
  const normalizedTag = tag?.toLowerCase().trim() ?? null;
  const variantSuffix = normalizedTag ? VARIANT_SUFFIX_MAP[normalizedTag] : undefined;
  const variantSuffixes = Object.values(VARIANT_SUFFIX_MAP);

  const matchesKey = (file: string) => {
    if (!normalizedKey) return true;
    const lower = file.toLowerCase();
    if (lower === normalizedKey) return true;
    if (lower.startsWith(`${normalizedKey}.`)) return true;
    return lower.startsWith(`${normalizedKey}_`);
  };

  const hasVariantSuffix = (file: string, suffix: string) => {
    const lower = file.toLowerCase();
    return lower.includes(`_${suffix}_`) || lower.endsWith(`_${suffix}.wav`) || lower.endsWith(`_${suffix}.mp3`) || lower.endsWith(`_${suffix}.flac`);
  };

  const isVariantFile = (file: string) => variantSuffixes.some((suffix) => hasVariantSuffix(file, suffix));

  const keyedFiles = audioFiles.filter(matchesKey);
  const baseCandidates = keyedFiles.filter((file) => !isVariantFile(file));
  const variantCandidates = variantSuffix ? keyedFiles.filter((file) => hasVariantSuffix(file, variantSuffix)) : [];

  const pool = variantCandidates.length > 0 ? variantCandidates : baseCandidates.length > 0 ? baseCandidates : keyedFiles.length > 0 ? keyedFiles : audioFiles;

  if (current && pool.includes(current)) {
    return current;
  }

  const candidates = pool.length ? pool : audioFiles;
  const source = `${seed}|${voice?.name ?? ""}|${styleName ?? ""}|${key ?? ""}|${normalizedTag ?? ""}`;
  let hash = 0;
  for (let i = 0; i < source.length; i += 1) {
    hash = (hash * 31 + source.charCodeAt(i)) | 0;
  }
  const index = Math.abs(hash) % candidates.length;
  return candidates[index] ?? current ?? null;
};

const toFileResults = (outputs?: FileResult[]) => outputs ?? [];

export const VoKitPanel = () => {
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

  const sampleStationsQuery = useQuery<SampleStationDescriptor[]>({
    queryKey: ["sample-stations"],
    queryFn: async () => (await api.get<SampleStationDescriptor[]>("/data/sample-stations")).data
  });

  const stationFormatsQuery = useQuery<StationFormatDescriptor[]>({
    queryKey: ["station-formats"],
    queryFn: async () => (await api.get<StationFormatDescriptor[]>("/data/station-formats")).data
  });

  const { apiKey } = useApiKey();
  const [scriptLines, setScriptLines] = useState<ScriptLine[]>([]);
  const [scriptName, setScriptName] = useState<string>("Loaded script");
  const [selectedVoice, setSelectedVoice] = useState<string | null>(null);
  const [selectedStyle, setSelectedStyle] = useState<string | null>(null);
  const [cloneVoice, setCloneVoice] = useState<string | null>(null);
  const [cloneSample, setCloneSample] = useState<string | null>(null);
  const [clonePitch, setClonePitch] = useState<number>(0);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [activeLoaderTab, setActiveLoaderTab] = useState<string>("upload");
  const [stationFormValues, setStationFormValues] = useState<Record<string, string>>({});
  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null);
  const [selectedSampleStationId, setSelectedSampleStationId] = useState<string | null>(null);
  const [batchJobId, setBatchJobId] = useState<string | null>(null);
  const [latestZipFile, setLatestZipFile] = useState<FileResult | null>(null);

  const triggerPlayback = (url?: string) => {
    if (!url) return;
    const withTs = url.includes('?') ? `${url}&ts=${Date.now()}` : `${url}?ts=${Date.now()}`;
    setPreviewUrl(withTs);
  };

  
  const defaults = defaultsQuery.data;
  const referenceVoices = referenceVoicesQuery.data ?? [];
  const cloneVoices = cloneVoicesQuery.data ?? [];

  const baseOptions = useMemo(() => (defaults ? defaultsToOptions(defaults) : null), [defaults]);

  useEffect(() => {
    if (!selectedVoice && referenceVoices.length > 0) {
      setSelectedVoice(referenceVoices[0].name);
    }
  }, [referenceVoices, selectedVoice]);

  useEffect(() => {
    if (selectedVoice) {
      const voice = referenceVoices.find((v) => v.name === selectedVoice);
      if (voice) {
        const style = voice.styles.find((s) => s.name === selectedStyle) ?? voice.styles[0];
        setSelectedStyle(style?.name ?? null);
      }
    }
  }, [referenceVoices, selectedVoice]);

  useEffect(() => {
    if (!selectedVoice || !selectedStyle) return;
    const voice = referenceVoices.find((v) => v.name === selectedVoice);
    if (!voice) return;
    setScriptLines((prev) => {
      let changed = false;
      const next = prev.map((line) => {
        const suggested = pickReferenceAudio(
          voice,
          selectedStyle,
          line.referenceKey,
          line.tag,
          line.referenceAudio,
          line.id
        );
        if (!line.referenceAudio || line.autoReference) {
          const shouldUpdate = !line.referenceAudio || suggested !== line.referenceAudio;
          if (shouldUpdate) changed = true;
          return { ...line, referenceAudio: suggested, autoReference: true };
        }
        return line;
      });
      return changed ? next : prev;
    });
  }, [selectedVoice, selectedStyle, referenceVoices, scriptLines]);

  const loadStationTemplate = async (templateId: string) => {
    const res = await api.get(`/data/station-formats/${templateId}`);
    const template = res.data as Record<string, { text: string; reference?: string }[]>;
    const keys = extractTemplateKeys(template);
    const existing: Record<string, string> = {};
    keys.forEach((key) => {
      existing[key] = stationFormValues[key] ?? "";
    });
    setStationFormValues(existing);
    setSelectedTemplateId(templateId);
    return template;
  };

  const applyStationSample = async (sampleId: string, template?: Record<string, any>) => {
    const res = await api.get(`/data/sample-stations/${encodeURIComponent(sampleId)}`);
    const sample = res.data as Record<string, string>;
    const values = { ...stationFormValues };
    Object.entries(sample).forEach(([key, value]) => {
      if (typeof value === "string") {
        values[key] = value;
      }
    });
    setStationFormValues(values);
    if (!selectedTemplateId) return;
    const tmpl = template ?? (await loadStationTemplate(selectedTemplateId));
    const lines = substituteTemplate(tmpl, values);
    setScriptLines(lines);
    setScriptName(`${sampleId} (${selectedTemplateId})`);
  };

  const analyzeMutation = useMutation({
    mutationFn: async (lines: string[]) => {
      const res = await api.post<AnalyzeResponse>("/scripts/analyze", { lines });
      return res.data.processed_lines;
    },
    onSuccess: (processed) => {
      setScriptLines((prev) => prev.map((line, index) => ({ ...line, text: processed[index] ?? line.text })));
    }
  });

  const singleLineMutation = useMutation({
    mutationFn: async (line: ScriptLine) => {
      if (!baseOptions) throw new Error("Defaults not loaded");
      if (!selectedVoice || !selectedStyle) throw new Error("Select a reference voice and style");
      if (!line.referenceAudio) throw new Error("Select reference audio");
      const voice = referenceVoices.find((v) => v.name === selectedVoice);
      const style = voice?.styles.find((s) => s.name === selectedStyle);
      const options = applyStyleOverrides({ ...baseOptions }, style, line.tag);
      options.sound_words_field = line.soundWordsField ?? baseOptions.sound_words_field ?? "";
      options.export_formats = Array.from(new Set([...options.export_formats, "wav"]));
      const payload = {
        line_id: line.id,
        text: line.text,
        section: line.section,
        reference_voice: selectedVoice,
        reference_style: selectedStyle,
        reference_audio: line.referenceAudio,
        tag: line.tag,
        sound_words_field: line.soundWordsField,
        clone_voice: cloneVoice ?? undefined,
        clone_audio: cloneSample ?? undefined,
        clone_pitch: clonePitch,
        options
      };
      const res = await api.post<LineGenerationResponse>("/lines/generate", payload);
      return res.data;
    },
    onSuccess: (response) => {
      setScriptLines((prev) =>
        prev.map((line) =>
          line.id === response.line_id
            ? {
                ...line,
                status: "completed",
                rawOutputs: toFileResults(response.raw_outputs),
                finalOutputs: toFileResults(response.final_outputs),
                error: null
              }
            : line
        )
      );
    },
    onError: (error, line) => {
      const message = error instanceof Error ? error.message : "Generation failed";
      setScriptLines((prev) => prev.map((item) => (item.id === line.id ? { ...item, status: "failed", error: message } : item)));
    }
  });

  const batchMutation = useMutation({
    mutationFn: async (lines: ScriptLine[]) => {
      if (!baseOptions) throw new Error("Defaults not loaded");
      if (!selectedVoice || !selectedStyle) throw new Error("Select a reference voice and style");
      const voice = referenceVoices.find((v) => v.name === selectedVoice);
      const style = voice?.styles.find((s) => s.name === selectedStyle);
      const payloadLines = lines.map((line) => {
        const options = applyStyleOverrides({ ...baseOptions }, style, line.tag);
        options.sound_words_field = line.soundWordsField ?? baseOptions.sound_words_field ?? "";
        options.export_formats = Array.from(new Set([...options.export_formats, "wav"]));
        if (!line.referenceAudio) {
          throw new Error(`Line ${line.id} is missing reference audio`);
        }
        return {
          line_id: line.id,
          text: line.text,
          section: line.section,
          reference_voice: selectedVoice,
          reference_style: selectedStyle,
          reference_audio: line.referenceAudio,
          tag: line.tag,
          sound_words_field: line.soundWordsField,
          clone_voice: cloneVoice ?? undefined,
          clone_audio: cloneSample ?? undefined,
          clone_pitch: clonePitch,
          options
        };
      });
      const payload: BatchCreateRequest = { lines: payloadLines, job_name: scriptName };
      const res = await api.post<{ job_id: string }>("/jobs", payload);
      return res.data.job_id;
    },
    onSuccess: (jobId) => {
      setLatestZipFile(null);
      setBatchJobId(jobId);
      setScriptLines((prev) => prev.map((line) => (line.status === "completed" ? line : { ...line, status: "processing", error: null })));
    }
  });

  const batchStatusQuery = useQuery<BatchJobStatus>({
    queryKey: ["job-status", batchJobId],
    queryFn: async () => (await api.get<BatchJobStatus>(`/jobs/${batchJobId}`)).data,
    enabled: Boolean(batchJobId),
    refetchInterval: 2000
  });

  useEffect(() => {
    const status = batchStatusQuery.data;
    if (!status) return;
    setScriptLines((prev) =>
      prev.map((line) => {
        const state = status.lines.find((l) => l.line_id === line.id);
        if (!state) return line;
        return {
          ...line,
          status: state.status,
          error: state.error ?? null,
          rawOutputs: state.raw_outputs ?? line.rawOutputs,
          finalOutputs: state.final_outputs ?? line.finalOutputs
        };
      })
    );
    if (["completed", "cancelled", "failed"].includes(status.state)) {
      setBatchJobId(null);
    }
    if (status.zip_file) {
      setLatestZipFile(status.zip_file);
    }
    if (status.state === "failed") {
      setLatestZipFile((prev) => (status.zip_file ? status.zip_file : prev));
    }
  }, [batchStatusQuery.data]);

  const cancelBatch = async () => {
    if (!batchJobId) return;
    await api.post(`/jobs/${batchJobId}/cancel`);
    setBatchJobId(null);
    batchStatusQuery.refetch();
  };

  const activeVoice = referenceVoices.find((voice) => voice.name === selectedVoice);
  const activeStyle = activeVoice?.styles.find((style) => style.name === selectedStyle);
  const tagOptions = [
    { value: "", label: "Default" },
    ...(activeStyle ? Object.keys(activeStyle.tag_settings).map((tag) => ({ value: tag, label: tag })) : [])
  ];

  const cloneVoiceOptions = cloneVoices.map((voice) => ({ value: voice.name, label: voice.name }));
  const cloneSampleOptions = cloneVoices
    .find((voice) => voice.name === cloneVoice)
    ?.files.map((file) => ({ value: file, label: file })) ?? [];

  const pendingLines = scriptLines.filter((line) => line.status !== "completed");

  const groupedSections = useMemo(() => {
    const map = new Map<string, ScriptLine[]>();
    scriptLines.forEach((line) => {
      if (!map.has(line.section)) {
        map.set(line.section, []);
      }
      map.get(line.section)!.push(line);
    });
    return Array.from(map.entries());
  }, [scriptLines]);

  const bundleZip = batchStatusQuery.data?.zip_file ?? latestZipFile;

  const buildReferenceUrl = (file: string | null) =>
    getReferenceUrl(selectedVoice, selectedStyle, file, apiKey);

  const handleFileUpload = async (file: File | null) => {
    if (!file) return;
    const text = await file.text();
    const parsed = parsePlainScript(text);
    setScriptLines(parsed);
    setScriptName(file.name);
  };

  const handleStationTemplateGenerate = async () => {
    if (!selectedTemplateId) return;
    const res = await api.get(`/data/station-formats/${selectedTemplateId}`);
    const template = res.data as Record<string, { text: string; reference?: string }[]>;
    const lines = substituteTemplate(template, stationFormValues);
    setScriptLines(lines);
    setScriptName(`${selectedTemplateId} template`);
  };

  const renderLoader = defaultsQuery.isLoading || referenceVoicesQuery.isLoading;

  return (
    <Stack gap="lg">
      <Card withBorder padding="lg" radius="md" shadow="sm">
        <Stack gap="sm">
          <Title order={4}>Script Source</Title>
          <Tabs value={activeLoaderTab} onChange={(value) => setActiveLoaderTab(value || "upload")}>
            <Tabs.List>
              <Tabs.Tab value="upload">Import .txt</Tabs.Tab>
              <Tabs.Tab value="format">Station format</Tabs.Tab>
            </Tabs.List>

            <Tabs.Panel value="upload" pt="md">
              <FileInput accept="text/plain" placeholder="Select .txt" onChange={handleFileUpload} />
            </Tabs.Panel>

            <Tabs.Panel value="format" pt="md">
              <Stack>
                <Group align="flex-end" gap="md">
                  <Select
                    label="Sample station"
                    placeholder="Optional"
                    data={sampleStationsQuery.data?.map((item) => ({ value: item.id, label: item.filename })) ?? []}
                    value={selectedSampleStationId}
                    onChange={(value) => {
                      setSelectedSampleStationId(value);
                      if (value && selectedTemplateId) {
                        loadStationTemplate(selectedTemplateId).then((template) => applyStationSample(value, template));
                      }
                    }}
                  />
                  <Select
                    label="Station format"
                    placeholder="Select"
                    data={stationFormatsQuery.data?.map((item) => ({ value: item.id, label: item.id })) ?? []}
                    value={selectedTemplateId}
                    onChange={(value) => {
                      setSelectedTemplateId(value);
                      if (value) loadStationTemplate(value);
                    }}
                  />
                  
                </Group>
                <Flex wrap="wrap" gap="sm">
                  {Object.entries(stationFormValues).map(([key, value]) => (
                    <TextInput
                      key={key}
                      label={key}
                      value={value}
                      onChange={(event: ChangeEvent<HTMLInputElement>) =>
                        setStationFormValues((prev) => ({ ...prev, [key]: event.currentTarget.value }))
                      }
                      className={classes.templateInput}
                    />
                  ))}
                </Flex>
                <Button onClick={handleStationTemplateGenerate} disabled={!selectedTemplateId}>
                  Generate from template
                </Button>
              </Stack>
            </Tabs.Panel>
          </Tabs>
          <Group>
            <Button
              variant="light"
              color="violet"
              onClick={() => analyzeMutation.mutateAsync(scriptLines.map((line) => line.text))}
              disabled={scriptLines.length === 0}
              loading={analyzeMutation.isLoading}
            >
              Analyze with ChatGPT
            </Button>
            {analyzeMutation.isError && (
              <Text c="red.4">Failed to analyze script.</Text>
            )}
          </Group>
        </Stack>
      </Card>

      <Card withBorder padding="lg" radius="md" shadow="sm">
        <Stack gap="md">
          <Title order={4}>Voice & Clone Settings</Title>
          {renderLoader ? (
            <Loader />
          ) : (
            <>
              <Group grow>
                <Select
                  label="Reference voice"
                  data={referenceVoices.map((voice) => ({ value: voice.name, label: voice.name }))}
                  value={selectedVoice}
                  onChange={setSelectedVoice}
                />
                <Select
                  label="Read style"
                  data={
                    activeVoice?.styles.map((style) => ({ value: style.name, label: style.name })) ?? []
                  }
                  value={selectedStyle}
                  onChange={setSelectedStyle}
                  disabled={!selectedVoice}
                />
              </Group>

              <Divider label="Clone (optional)" labelPosition="left" />
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
              <Stack gap={4}>
                <Text fw={500}>Clone pitch ({clonePitch} semitones)</Text>
                <Slider min={-12} max={12} step={1} value={clonePitch} onChange={setClonePitch} marks={[{ value: 0, label: "0" }]} />
              </Stack>
            </>
          )}
        </Stack>
      </Card>

      <Card withBorder padding="lg" radius="md" shadow="sm">
        <Group align="center" justify="space-between">
          <Title order={4}>{scriptName}</Title>
          <Group>
            <Button
              variant="light"
              color="gray"
              onClick={() => {
                setScriptLines((prev) => prev.map((line) => ({ ...line, status: "pending", rawOutputs: undefined, finalOutputs: undefined, error: null })));
              }}
            >
              Reset statuses
            </Button>
            <Button
              color="violet"
              onClick={() => batchMutation.mutateAsync(scriptLines.filter((line) => line.status !== "completed"))}
              disabled={scriptLines.length === 0 || pendingLines.length === 0 || !baseOptions || !selectedVoice || !selectedStyle}
              loading={batchMutation.isLoading}
            >
              Generate all pending ({pendingLines.length})
            </Button>
            {batchJobId && (
              <Button color="red" variant="outline" onClick={cancelBatch}>
                Cancel batch
              </Button>
            )}
          </Group>
        </Group>
        <Space h="md" />

        {bundleZip && (
          <Alert color="teal" radius="sm" title="Bundle ready" icon={<IconChecks size={16} />}>
            <Group gap="sm" align="center">
              <Text>Download the stitched bundle:</Text>
              <Button component="a" href={bundleZip.url ?? undefined} variant="light" size="xs" disabled={!bundleZip.url}>
                Download ZIP
              </Button>
            </Group>
          </Alert>
        )}

        <Space h="sm" />
        <div className={classes.tableWrapper}>
          <Table highlightOnHover verticalSpacing="sm">
            <Table.Thead>
              <Table.Tr>
                <Table.Th className={classes.textCol}>Text</Table.Th>
                <Table.Th className={classes.tagCol}>Tag</Table.Th>
                <Table.Th className={classes.refCol}>Reference Audio</Table.Th>
                <Table.Th className={classes.statusCol}>Status</Table.Th>
                <Table.Th className={classes.actionsCol}>Actions</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {groupedSections.map(([section, lines]) => (
                <Fragment key={section}>
                  <Table.Tr className={classes.sectionRow}>
                    <Table.Td colSpan={5}>
                      <Text fw={600} tt="uppercase" size="sm">
                        {section}
                      </Text>
                    </Table.Td>
                  </Table.Tr>
                  {lines.map((line) => {
                    const mp3 = line.finalOutputs?.find((file) => file.path.endsWith(".mp3"));
                    return (
                      <Table.Tr key={line.id}>
                        <Table.Td className={classes.textCol}>
                          <Textarea
                            autosize
                            minRows={2}
                            value={line.text}
                            onChange={(event) =>
                              setScriptLines((prev) =>
                                prev.map((item) => (item.id === line.id ? { ...item, text: event.currentTarget.value } : item))
                              )
                            }
                          />
                        </Table.Td>
                        <Table.Td className={classes.tagCol}>
                          <Select
                            value={line.tag ?? ""}
                            onChange={(value) =>
                              setScriptLines((prev) =>
                                prev.map((item) => (item.id === line.id ? { ...item, tag: value || null } : item))
                              )
                            }
                            data={tagOptions}
                          />
                        </Table.Td>
                        <Table.Td className={classes.refCol}>
                          <Stack gap={4}>
                            <Select
                              searchable
                              clearable
                              placeholder="Select"
                              value={line.referenceAudio}
                              data={activeStyle?.audio_files.map((file) => ({ value: file, label: file })) ?? []}
                              onChange={(value) =>
                                setScriptLines((prev) =>
                                  prev.map((item) =>
                                    item.id === line.id
                                      ? { ...item, referenceAudio: value, autoReference: value ? false : item.autoReference }
                                      : item
                                  )
                                )
                              }
                            />
                            <PlayButton url={buildReferenceUrl(line.referenceAudio ?? null)} onPlay={triggerPlayback} label="Preview" />
                          </Stack>
                        </Table.Td>
                        <Table.Td className={classes.statusCol}>
                          <StatusBadge status={line.status} error={line.error} />
                        </Table.Td>
                        <Table.Td className={classes.actionsCol}>
                          <Stack gap={6}>
                            <Button
                              size="xs"
                              color="violet"
                              onClick={() => {
                                setScriptLines((prev) =>
                                  prev.map((item) => (item.id === line.id ? { ...item, status: "processing", error: null } : item))
                                );
                                singleLineMutation.mutate(line);
                              }}
                              loading={singleLineMutation.isLoading && singleLineMutation.variables?.id === line.id}
                              disabled={!baseOptions}
                            >
                              Generate
                            </Button>
                            <Group gap="xs">
                              <ActionIcon
                                variant="light"
                                size="md"
                                radius="md"
                                color="grape"
                                disabled={!mp3?.url || line.status === "processing"}
                                onClick={() => mp3?.url && triggerPlayback(mp3.url)}
                              >
                                <IconPlayerPlay size={16} />
                              </ActionIcon>
                              <ActionIcon
                                variant="light"
                                size="md"
                                radius="md"
                                color="grape"
                                component="a"
                                href={mp3?.url ?? undefined}
                                download
                                disabled={!mp3?.url || line.status === "processing"}
                              >
                                <IconDownload size={16} />
                              </ActionIcon>
                            </Group>
                          </Stack>
                        </Table.Td>
                      </Table.Tr>
                    );
                  })}
                </Fragment>
              ))}
            </Table.Tbody>
          </Table>
        </div>
      </Card>

      {batchStatusQuery.isFetching && batchJobId && (
        <Alert color="violet" icon={<Loader size="xs" />}>
          Processing batch… {Math.round((batchStatusQuery.data?.progress ?? 0) * 100)}%
        </Alert>
      )}
      {previewUrl && (
        <audio key={previewUrl} src={previewUrl} autoPlay onEnded={() => setPreviewUrl(null)} style={{ display: "none" }} />
      )}
    </Stack>
  );
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
  label = "Play",
  disabled = false
}: {
  url?: string;
  onPlay?: (url: string) => void;
  label?: string;
  disabled?: boolean;
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

const StatusBadge = ({ status, error }: { status: LineStatus; error?: string | null }) => {
  const map: Record<LineStatus, { color: string; label: string }> = {
    pending: { color: "gray", label: "Pending" },
    processing: { color: "yellow", label: "Processing" },
    completed: { color: "green", label: "Completed" },
    failed: { color: "red", label: "Failed" },
    cancelled: { color: "orange", label: "Cancelled" }
  };
  const item = map[status];
  return (
    <Stack gap={4}>
      <Badge color={item.color}>{item.label}</Badge>
      {error ? (
        <Text size="xs" c="red.4">
          {error}
        </Text>
      ) : null}
    </Stack>
  );
};
