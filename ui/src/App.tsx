import { useState } from "react";
import {
  AppShell,
  TextInput,
  Button,
  Group,
  Tabs,
  Container,
  Title,
  Alert
} from "@mantine/core";
import { IconAlertCircle } from "@tabler/icons-react";
import { useApiKey } from "@/hooks/useApiKey";
import { VoiceClonePanel } from "@/components/VoiceClonePanel";
import { VoKitPanel } from "@/components/VoKitPanel";

export default function App() {
  const { apiKey, setApiKey } = useApiKey();
  const [candidateKey, setCandidateKey] = useState(apiKey ?? "");
  const [activeTab, setActiveTab] = useState<string>("vo-kit");

  const applyApiKey = () => {
    setApiKey(candidateKey.trim() || null);
  };

  const clearKey = () => {
    setCandidateKey("");
    setApiKey(null);
  };

  return (
    <AppShell
      header={{ height: 70 }}
      padding="md"
      styles={{
        main: {
          background: "linear-gradient(180deg,#0f172a 0%,#020617 100%)",
          color: "white"
        }
      }}
    >
      <AppShell.Header withBorder={false}>
        <Container size="lg" h="100%">
          <Group align="center" justify="space-between" h="100%">
            <Title order={3} c="violet.2">
              Chatterbox Studio
            </Title>
            <Group gap="xs">
              <TextInput
                placeholder="Enter API key"
                value={candidateKey}
                onChange={(event) => setCandidateKey(event.currentTarget.value)}
                type="password"
                style={{ minWidth: 260 }}
              />
              <Button onClick={applyApiKey} color="violet">
                Connect
              </Button>
              {apiKey ? (
                <Button variant="outline" color="gray" onClick={clearKey}>
                  Disconnect
                </Button>
              ) : null}
            </Group>
          </Group>
        </Container>
      </AppShell.Header>

      <AppShell.Main>
        <Container size="lg" py="xl">
          {apiKey ? (
            <Tabs value={activeTab} onChange={(val) => setActiveTab(val || "vo-kit")}>\
              <Tabs.List>
                <Tabs.Tab value="vo-kit">VO Kit</Tabs.Tab>
                <Tabs.Tab value="voice-clone">Voice Clone</Tabs.Tab>
              </Tabs.List>

              <Tabs.Panel value="vo-kit" pt="md">
                <VoKitPanel />
              </Tabs.Panel>

              <Tabs.Panel value="voice-clone" pt="md">
                <VoiceClonePanel />
              </Tabs.Panel>
            </Tabs>
          ) : (
            <Alert
              variant="light"
              color="violet"
              title="API key required"
              icon={<IconAlertCircle size={16} />}
            >
              Enter the shared internal password. The UI stores it in local storage so you only need to
              provide it once per browser.
            </Alert>
          )}
        </Container>
      </AppShell.Main>
    </AppShell>
  );
}
