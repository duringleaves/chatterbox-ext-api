import { useEffect, useRef, useState } from "react";
import { Alert, Button, Group, Stack, Text } from "@mantine/core";
import { IconAlertCircle, IconMicrophone, IconPlayerStop, IconRefresh } from "@tabler/icons-react";

type AudioRecorderProps = {
  onChange: (file: File | null) => void;
};

export const AudioRecorder = ({ onChange }: AudioRecorderProps) => {
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const [isRecording, setIsRecording] = useState(false);
  const [recordedUrl, setRecordedUrl] = useState<string | null>(null);
  const [recordedFileName, setRecordedFileName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const cleanupStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  };

  useEffect(() => {
    return () => {
      cleanupStream();
      if (recorderRef.current && recorderRef.current.state !== "inactive") {
        recorderRef.current.stop();
      }
      if (recordedUrl) {
        URL.revokeObjectURL(recordedUrl);
      }
    };
  }, [recordedUrl]);

  const handleStop = () => {
    const chunks = chunksRef.current;
    chunksRef.current = [];
    const mimeType = recorderRef.current?.mimeType || "audio/webm";
    const blob = new Blob(chunks, { type: mimeType });
    const extension = mimeType.includes("mp4") ? "m4a" : "webm";
    const fileName = `recording-${Date.now()}.${extension}`;
    const file = new File([blob], fileName, { type: mimeType });
    const url = URL.createObjectURL(blob);

    if (recordedUrl) {
      URL.revokeObjectURL(recordedUrl);
    }

    setRecordedUrl(url);
    setRecordedFileName(fileName);
    setIsRecording(false);
    onChange(file);
  };

  const startRecording = async () => {
    if (typeof window === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      setError("Recording is not supported in this browser");
      return;
    }

    if (!("MediaRecorder" in window)) {
      setError("Recording is not supported in this browser");
      return;
    }

    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const preferredTypes = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4",
        "audio/mpeg"
      ];
      const selectedType = preferredTypes.find((type) => MediaRecorder.isTypeSupported(type)) || "";

      const recorder = new MediaRecorder(stream, selectedType ? { mimeType: selectedType } : undefined);
      recorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = handleStop;
      recorder.onerror = (event) => {
        setError(event.error?.message || "Recording error occurred");
        setIsRecording(false);
        cleanupStream();
      };

      if (recordedUrl) {
        URL.revokeObjectURL(recordedUrl);
        setRecordedUrl(null);
        setRecordedFileName(null);
      }

      onChange(null);
      recorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Failed to start recording", err);
      setError("Microphone access was denied or unavailable");
      cleanupStream();
    }
  };

  const stopRecording = () => {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    cleanupStream();
  };

  const resetRecording = () => {
    if (recordedUrl) {
      URL.revokeObjectURL(recordedUrl);
    }
    setRecordedUrl(null);
    setRecordedFileName(null);
    setIsRecording(false);
    chunksRef.current = [];
    onChange(null);
  };

  return (
    <Stack>
      <Group>
        <Button
          leftSection={<IconMicrophone size={16} />}
          onClick={startRecording}
          disabled={isRecording}
        >
          Start recording
        </Button>
        <Button
          color="red"
          leftSection={<IconPlayerStop size={16} />}
          onClick={stopRecording}
          disabled={!isRecording}
        >
          Stop
        </Button>
        <Button
          variant="light"
          color="gray"
          leftSection={<IconRefresh size={16} />}
          onClick={resetRecording}
          disabled={isRecording || !recordedUrl}
        >
          Reset
        </Button>
      </Group>

      {isRecording && <Text c="dimmed">Recording… press Stop when you are done.</Text>}

      {recordedUrl && (
        <Stack gap={4}>
          <Text fw={500}>{recordedFileName}</Text>
          <audio controls src={recordedUrl} />
        </Stack>
      )}

      {error && (
        <Alert color="red" icon={<IconAlertCircle size={16} />}>
          {error}
        </Alert>
      )}
    </Stack>
  );
};
