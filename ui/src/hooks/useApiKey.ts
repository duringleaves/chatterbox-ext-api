import { useEffect, useState } from "react";
import { setApiKey } from "@/lib/api";

const LOCAL_STORAGE_KEY = "chatterbox-api-key";

export const useApiKey = () => {
  const [apiKey, setApiKeyState] = useState<string | null>(() => {
    return localStorage.getItem(LOCAL_STORAGE_KEY);
  });

  useEffect(() => {
    setApiKey(apiKey);
    if (apiKey) {
      localStorage.setItem(LOCAL_STORAGE_KEY, apiKey);
    } else {
      localStorage.removeItem(LOCAL_STORAGE_KEY);
    }
  }, [apiKey]);

  return { apiKey, setApiKey: setApiKeyState } as const;
};
