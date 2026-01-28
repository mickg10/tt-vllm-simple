import { useState, useEffect, useCallback } from 'react';

export interface Device {
  id: string;
  board_type: string;
  pci_index: number;
  pci_bdf: string;
  status: string;
  temperature: number;
  power: number;
  voltage: number;
  current: number;
  aiclk: number;
  arcclk: number;
  firmware: string;
  serial: string;
}

export interface DevicesResponse {
  timestamp: string;
  device_count: number;
  devices: Device[];
  totals: {
    power: number;
    avg_temperature: number;
  };
}

export interface VLLMMetrics {
  timestamp: string;
  available: boolean;
  model_name: string;
  requests_running: number;
  requests_waiting: number;
  gpu_cache_usage_percent: number;
  prompt_tokens_total: number;
  generation_tokens_total: number;
  avg_time_to_first_token_ms: number;
  avg_time_per_output_token_ms: number;
  avg_e2e_latency_s: number;
  error?: string;
}

export interface HistoryPoint {
  timestamp: number;
  temperature?: number;
  power?: number;
  voltage?: number;
  current?: number;
  aiclk?: number;
  arcclk?: number;
}

export interface VLLMHistoryPoint {
  timestamp: number;
  model_name: string;
  requests_running: number;
  requests_waiting: number;
  gpu_cache_usage: number;
  prompt_tokens_total: number;
  generation_tokens_total: number;
  avg_ttft: number;
  avg_tpot: number;
  avg_e2e_latency: number;
}

export interface DeviceHistoryResponse {
  time_range: string;
  devices: Record<string, HistoryPoint[]>;
}

export interface VLLMHistoryResponse {
  time_range: string;
  data_points: number;
  history: VLLMHistoryPoint[];
}

export type TimeRange = '10m' | '1h' | '6h' | '24h' | '1w';

const API_BASE = '';

export function useDevices(refreshInterval = 5000) {
  const [data, setData] = useState<DevicesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchDevices = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/devices`);
      if (!response.ok) throw new Error('Failed to fetch devices');
      const json = await response.json();
      setData(json);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDevices();
    const interval = setInterval(fetchDevices, refreshInterval);
    return () => clearInterval(interval);
  }, [fetchDevices, refreshInterval]);

  return { data, error, loading, refetch: fetchDevices };
}

export function useVLLM(refreshInterval = 5000) {
  const [data, setData] = useState<VLLMMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchVLLM = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/vllm`);
      if (!response.ok) throw new Error('Failed to fetch vLLM metrics');
      const json = await response.json();
      setData(json);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchVLLM();
    const interval = setInterval(fetchVLLM, refreshInterval);
    return () => clearInterval(interval);
  }, [fetchVLLM, refreshInterval]);

  return { data, error, loading, refetch: fetchVLLM };
}

export function useDeviceHistory(timeRange: TimeRange, refreshInterval = 10000) {
  const [data, setData] = useState<DeviceHistoryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchHistory = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/history/devices?range=${timeRange}`);
      if (!response.ok) throw new Error('Failed to fetch device history');
      const json = await response.json();
      setData(json);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [timeRange]);

  useEffect(() => {
    fetchHistory();
    const interval = setInterval(fetchHistory, refreshInterval);
    return () => clearInterval(interval);
  }, [fetchHistory, refreshInterval]);

  return { data, error, loading, refetch: fetchHistory };
}

export function useVLLMHistory(timeRange: TimeRange, refreshInterval = 10000) {
  const [data, setData] = useState<VLLMHistoryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchHistory = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/history/vllm?range=${timeRange}`);
      if (!response.ok) throw new Error('Failed to fetch vLLM history');
      const json = await response.json();
      setData(json);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [timeRange]);

  useEffect(() => {
    fetchHistory();
    const interval = setInterval(fetchHistory, refreshInterval);
    return () => clearInterval(interval);
  }, [fetchHistory, refreshInterval]);

  return { data, error, loading, refetch: fetchHistory };
}
