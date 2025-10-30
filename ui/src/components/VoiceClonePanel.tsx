import { useCallback, useEffect, useMemo, useState } from "react";
import { Alert, Button, Card, FileInput, Group, Loader, NumberInput, Select, Space, Stack, Table, Text, Title, Switch } from "@mantine/core";
import { useQuery, useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { CloneVoice, FileResult } from "@/lib/types";
import { IconAlertCircle, IconMicrophone, IconUpload } from "@tabler/icons-react";
import { AudioRecorder } from "./AudioRecorder";

const toBase64 = (file: File) =>
  new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result;
      if (typeof result === "string") {
        const base64 = result.split(",").pop() || "";
        resolve(base64);
      } else {
        reject(new Error("Failed to read file"));
      }
    };
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsDataURL(file);
  });

type VoiceCloneOutput = FileResult & { renderedAt: number };

export const VoiceClonePanel = () => {
  const { data: cloneVoices, isLoading } = useQuery<CloneVoice[]>({
    queryKey: ["clone-voices"],
    queryFn: async () => {
      const res = await api.get<CloneVoice[]>("/voices/clone");
      return res.data;
    }
  });

  const [inputFile, setInputFile] = useState<File | null>(null);
  const [selectedVoice, setSelectedVoice] = useState<string | null>(null);
  const [voiceSettings, setVoiceSettings] = useState<Record<string, any> | null>(null);
  const [modelId, setModelId] = useState<string>("eleven_multilingual_sts_v2");
  const [outputs, setOutputs] = useState<VoiceCloneOutput[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [inputMode, setInputMode] = useState<"upload" | "record">("upload");
  const [cloneSettingsOpen, setCloneSettingsOpen] = useState<boolean>(false);

  const voiceOptions = useMemo(
    () => (cloneVoices ?? []).map((voice) => ({ value: voice.id, label: voice.name })),
    [cloneVoices]
  );
  const selectedVoiceEntry = useMemo(
    () => (cloneVoices ?? []).find((voice) => voice.id === selectedVoice) ?? null,
    [cloneVoices, selectedVoice]
  );
  const modelOptions = useMemo(
    () => [
      { value: "eleven_multilingual_sts_v2", label: "Multilingual" },
      { value: "eleven_english_sts_v2", label: "English" }
    ],
    []
  );
  const settingEntries = useMemo(() => {
    if (!selectedVoiceEntry) return [];
    const source = voiceSettings ?? selectedVoiceEntry.voice_settings ?? {};
    return Object.entries(source).sort((a, b) => a[0].localeCompare(b[0]));
  }, [selectedVoiceEntry, voiceSettings]);

  useEffect(() => {
    if (!selectedVoice) {
      setVoiceSettings(null);
      setModelId("eleven_multilingual_sts_v2");
      return;
    }
    const voice = (cloneVoices ?? []).find((entry) => entry.id === selectedVoice);
    if (!voice) {
      setVoiceSettings(null);
      return;
    }
    setVoiceSettings({ ...(voice.voice_settings ?? {}) });
  }, [selectedVoice, cloneVoices]);

  const handleSettingChange = useCallback(
    (key: string, value: number | boolean) => {
      setVoiceSettings((prev) => {
        const base = prev ?? { ...(selectedVoiceEntry?.voice_settings ?? {}) };
        return { ...base, [key]: value };
      });
    },
    [selectedVoiceEntry]
  );

  const resetVoiceSettings = useCallback(() => {
    if (!selectedVoiceEntry) {
      setVoiceSettings(null);
      return;
    }
    setVoiceSettings({ ...(selectedVoiceEntry.voice_settings ?? {}) });
  }, [selectedVoiceEntry]);

  useEffect(() => {
    if (!selectedVoice) {
      setCloneSettingsOpen(false);
    }
  }, [selectedVoice]);

  const mutation = useMutation({
    mutationFn: async () => {
      if (!inputFile) throw new Error("Please provide an input audio file");
      if (!selectedVoiceEntry) throw new Error("Select a clone voice");
      const data = await toBase64(inputFile);
      const settingsPayload = voiceSettings ?? selectedVoiceEntry.voice_settings ?? {};
      const payload = {
        input_audio: {
          filename: inputFile.name,
          data
        },
        voice_id: selectedVoiceEntry.voice_id,
        model_id: modelId,
        voice_settings: Object.keys(settingsPayload).length ? settingsPayload : undefined,
        export_formats: ["mp3", "wav"],
        return_audio_base64: false
      };
      const res = await api.post<{ outputs: FileResult[] }>("/voice/convert", payload);
      return res.data.outputs;
    },
    onSuccess: (data) => {
      if (!data.length) {
        setError("No audio output was returned from the server.");
        setOutputs([]);
        return;
      }
      const timestamp = Date.now();
      setOutputs(
        data.map((file) => ({
          ...file,
          renderedAt: timestamp
        }))
      );
      setError(null);
    },
    onError: (err: unknown) => {
      const message = err instanceof Error ? err.message : "Failed to render voice clone";
      setError(message);
      setOutputs([]);
    }
  });

  return (
    <Stack gap="md">
      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack>
          <Title order={4}>Quick Voice Clone</Title>
          <Text c="dimmed">
            Upload or record an audio clip, pick a destination voice, and the API will render a cloned take.
          </Text>

          <Stack gap="sm">
            <Group>
              <Button
                leftSection={<IconUpload size={16} />}
                variant={inputMode === "upload" ? "filled" : "light"}
                color="violet"
                onClick={() => {
                  if (inputMode !== "upload") {
                    setInputMode("upload");
                    setInputFile(null);
                  }
                }}
              >
                Upload file
              </Button>
              <Button
                leftSection={<IconMicrophone size={16} />}
                variant={inputMode === "record" ? "filled" : "light"}
                color="violet"
                onClick={() => {
                  if (inputMode !== "record") {
                    setInputMode("record");
                    setInputFile(null);
                  }
                }}
              >
                Record audio
              </Button>
            </Group>

            {inputMode === "upload" ? (
              <FileInput
                label="Upload audio to clone"
                placeholder="Select a WAV/MP3/FLAC file"
                value={inputFile}
                onChange={setInputFile}
                accept="audio/*"
                withAsterisk
              />
            ) : (
              <Stack gap="xs">
                <Text size="sm" c="dimmed">
                  Press record to capture a short clip with your microphone. The recording stays local and is
                  sent as the input file for cloning.
                </Text>
                <AudioRecorder key={inputMode} onChange={setInputFile} />
              </Stack>
            )}
          </Stack>

          <Stack gap="sm">
            <Group justify="space-between" align="flex-end">
              <Select
                label="Clone voice"
                placeholder={isLoading ? "Loading voices..." : "Select voice"}
                data={voiceOptions}
                value={selectedVoice}
                onChange={(value) => {
                  setSelectedVoice(value ?? null);
                }}
                withAsterisk
                nothingFound="No clone voices"
                style={{ flexGrow: 1 }}
              />
              <Button
                variant="subtle"
                size="xs"
                color="violet"
                onClick={() => setCloneSettingsOpen((prev) => !prev)}
                disabled={!selectedVoice}
              >
                {cloneSettingsOpen ? "Hide settings" : "Show settings"}
              </Button>
            </Group>
            {cloneSettingsOpen && (
              <Stack gap="sm">
                {selectedVoiceEntry?.description && (
                  <Text size="sm" c="dimmed">
                    {selectedVoiceEntry.description}
                  </Text>
                )}
                <Select
                  label="Model"
                  placeholder="Choose ElevenLabs model"
                  data={modelOptions}
                  value={modelId}
                  onChange={(value) => setModelId(value ?? "eleven_multilingual_sts_v2")}
                  disabled={!selectedVoice}
                />
                <Stack gap="xs">
                  <Group justify="space-between" align="center">
                    <Text size="sm" fw={500}>
                      Voice settings
                    </Text>
                    <Button variant="subtle" size="xs" onClick={resetVoiceSettings} disabled={!selectedVoiceEntry}>
                      Reset to defaults
                    </Button>
                  </Group>
                  {settingEntries.length > 0 ? (
                    settingEntries.map(([key, value]) => {
                      if (typeof value === "boolean") {
                        return (
                          <Switch
                            key={key}
                            label={key}
                            checked={Boolean(value)}
                            onChange={(event) => handleSettingChange(key, event.currentTarget.checked)}
                          />
                        );
                      }
                      if (typeof value === "number") {
                        return (
                          <NumberInput
                            key={key}
                            label={key}
                            value={value}
                            onChange={(val) => {
                              const numeric = typeof val === "number" ? val : Number(val);
                              if (!Number.isNaN(numeric)) handleSettingChange(key, numeric);
                            }}
                            step={0.05}
                            precision={2}
                          />
                        );
                      }
                      return (
                        <Text key={key} size="sm">
                          {key}: {String(value)}
                        </Text>
                      );
                    })
                  ) : (
                    <Text size="sm" c="dimmed">
                      This voice has no adjustable settings.
                    </Text>
                  )}
                </Stack>
              </Stack>
            )}
          </Stack>

          <Group>
            <Button
              size="md"
              color="violet"
              onClick={() => mutation.mutate()}
              loading={mutation.isPending}
              disabled={!inputFile || !selectedVoice || mutation.isPending}
            >
              Render Clone
            </Button>
            {mutation.isPending && <Loader size="sm" />}
          </Group>
          {error && (
            <Alert color="red" icon={<IconAlertCircle size={16} />}>
              {error}
            </Alert>
          )}
        </Stack>
      </Card>

      {outputs.length > 0 && (
        <Card withBorder shadow="sm" radius="md" padding="lg">
          <Title order={5}>Rendered Outputs</Title>
          <Space h="sm" />
          <Table highlightOnHover>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>File</Table.Th>
                <Table.Th>Rendered</Table.Th>
                <Table.Th>Preview</Table.Th>
                <Table.Th>Download</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {outputs.map((file) => (
                <Table.Tr key={`${file.path}-${file.renderedAt}`}>
                  <Table.Td>{file.path}</Table.Td>
                  <Table.Td>{new Date(file.renderedAt).toLocaleString()}</Table.Td>
                  <Table.Td>
                    {file.url ? <audio controls src={file.url} style={{ width: 220 }} /> : <Text c="dimmed">No URL</Text>}
                  </Table.Td>
                  <Table.Td>
                    {file.url ? (
                      <Button component="a" href={file.url} download variant="light" size="xs">
                        Download
                      </Button>
                    ) : (
                      <Text c="dimmed">—</Text>
                    )}
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Card>
      )}

    </Stack>
  );
};
