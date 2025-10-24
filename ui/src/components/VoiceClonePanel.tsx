import { useState } from "react";
import {
  Alert,
  Button,
  Card,
  FileInput,
  Group,
  Loader,
  Select,
  Space,
  Stack,
  Table,
  Text,
  Title
} from "@mantine/core";
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
  const [selectedSample, setSelectedSample] = useState<string | null>(null);
  const [outputs, setOutputs] = useState<VoiceCloneOutput[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [inputMode, setInputMode] = useState<"upload" | "record">("upload");
  const pitch = 0;

  const mutation = useMutation({
    mutationFn: async () => {
      if (!inputFile) throw new Error("Please provide an input audio file");
      if (!selectedVoice) throw new Error("Select a clone voice");
      const sample = selectedSample || cloneVoices?.find((v) => v.name === selectedVoice)?.files?.[0];
      if (!sample) throw new Error("No sample available for the selected voice");
      const data = await toBase64(inputFile);
      const payload = {
        input_audio: {
          filename: inputFile.name,
          data
        },
        target_voice_audio: {
          filename: sample,
          path: `data/clone_voices/${selectedVoice}/${sample}`
        },
        pitch_shift: pitch,
        disable_watermark: true,
        export_formats: ["wav"],
        return_audio_base64: false
      };
      const res = await api.post<{ outputs: FileResult[] }>("/voice/convert", payload);
      return res.data.outputs;
    },
    onSuccess: (data) => {
      const timestamp = Date.now();
      const wavOutputs = data.filter((file) => file.path.toLowerCase().endsWith(".wav"));
      setOutputs((prev) => [
        ...prev,
        ...wavOutputs.map((file) => ({
          ...file,
          renderedAt: timestamp
        }))
      ]);
      setError(null);
    },
    onError: (err: unknown) => {
      const message = err instanceof Error ? err.message : "Failed to render voice clone";
      setError(message);
    }
  });

  const voiceOptions = (cloneVoices || []).map((voice) => ({ value: voice.name, label: voice.name }));
  const sampleOptions = cloneVoices
    ?.find((voice) => voice.name === selectedVoice)
    ?.files.map((file) => ({ value: file, label: file })) ?? [];

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

          <Group grow>
            <Select
              label="Clone voice"
              placeholder={isLoading ? "Loading voices..." : "Select voice"}
              data={voiceOptions}
              value={selectedVoice}
              onChange={(value) => {
                setSelectedVoice(value);
                setSelectedSample(null);
              }}
              withAsterisk
            />
            <Select
              label="Reference sample"
              placeholder="Auto select"
              data={sampleOptions}
              value={selectedSample}
              onChange={setSelectedSample}
              disabled={!selectedVoice}
            />
          </Group>

          <Stack gap={4}>
            <Text fw={500}>Pitch shift ({pitch} semitones)</Text>
            <Slider min={-12} max={12} step={1} value={pitch} onChange={setPitch} marks={[{ value: 0, label: "0" }]} />
          </Stack>

          <Group>
            <Button
              size="md"
              color="violet"
              onClick={() => mutation.mutateAsync()}
              loading={mutation.isLoading}
              disabled={!inputFile || !selectedVoice || mutation.isLoading}
            >
              Render Clone
            </Button>
            {mutation.isLoading && <Loader size="sm" />}
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
