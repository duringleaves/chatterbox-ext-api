import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export const api = axios.create({
  baseURL: API_BASE || "/",
  withCredentials: false
});

export const setApiKey = (token: string | null) => {
  if (token) {
    api.defaults.headers.common["Authorization"] = `Bearer ${token}`;
  } else {
    delete api.defaults.headers.common["Authorization"];
  }
};

