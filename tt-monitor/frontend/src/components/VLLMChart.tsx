import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';
import { VLLMHistoryResponse } from '../hooks/useDevices';

interface Props {
  history: VLLMHistoryResponse;
}

export function VLLMChart({ history }: Props) {
  if (!history.history || history.history.length === 0) {
    return <div className="no-data">No vLLM history data available</div>;
  }

  // Transform data for charts
  const chartData = history.history.map((point) => ({
    time: new Date(point.timestamp * 1000).toLocaleTimeString(),
    timestamp: point.timestamp,
    requests_running: point.requests_running,
    requests_waiting: point.requests_waiting,
    cache_usage: point.gpu_cache_usage * 100,  // Already percentage from backend
    ttft_ms: point.avg_ttft * 1000,
    tpot_ms: point.avg_tpot * 1000,
    tokens_per_sec: point.avg_tpot > 0 ? 1 / point.avg_tpot : 0,
    prompt_tokens: point.prompt_tokens_total,
    gen_tokens: point.generation_tokens_total,
  }));

  // Calculate tokens/sec delta for throughput chart
  const throughputData = chartData.map((point, idx) => {
    if (idx === 0) return { ...point, throughput: 0 };
    const prev = chartData[idx - 1];
    const timeDelta = point.timestamp - prev.timestamp;
    const tokenDelta = point.gen_tokens - prev.gen_tokens;
    return {
      ...point,
      throughput: timeDelta > 0 ? tokenDelta / timeDelta : 0,
    };
  }).slice(1);  // Remove first point (no delta)

  return (
    <>
      <div className="chart-container">
        <h3 style={{ fontSize: '14px', color: '#71767b', marginBottom: '12px' }}>
          Throughput (tokens/sec)
        </h3>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={throughputData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2f3336" />
            <XAxis dataKey="time" stroke="#71767b" fontSize={11} tickLine={false} />
            <YAxis stroke="#71767b" fontSize={11} tickLine={false} domain={[0, 'auto']} />
            <Tooltip
              contentStyle={{
                background: '#16181c',
                border: '1px solid #2f3336',
                borderRadius: '8px',
              }}
              formatter={(value: number) => [`${value.toFixed(1)} tok/s`, 'Throughput']}
            />
            <Area
              type="monotone"
              dataKey="throughput"
              stroke="#00ba7c"
              fill="#00ba7c"
              fillOpacity={0.3}
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-container">
        <h3 style={{ fontSize: '14px', color: '#71767b', marginBottom: '12px' }}>
          Requests & KV Cache
        </h3>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2f3336" />
            <XAxis dataKey="time" stroke="#71767b" fontSize={11} tickLine={false} />
            <YAxis yAxisId="left" stroke="#71767b" fontSize={11} tickLine={false} />
            <YAxis yAxisId="right" orientation="right" stroke="#71767b" fontSize={11} tickLine={false} domain={[0, 100]} />
            <Tooltip
              contentStyle={{
                background: '#16181c',
                border: '1px solid #2f3336',
                borderRadius: '8px',
              }}
            />
            <Legend />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="requests_running"
              stroke="#1d9bf0"
              strokeWidth={2}
              dot={false}
              name="Running"
            />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="requests_waiting"
              stroke="#f91880"
              strokeWidth={2}
              dot={false}
              name="Waiting"
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="cache_usage"
              stroke="#ffa500"
              strokeWidth={2}
              dot={false}
              name="KV Cache %"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-container">
        <h3 style={{ fontSize: '14px', color: '#71767b', marginBottom: '12px' }}>
          Latency (TTFT)
        </h3>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2f3336" />
            <XAxis dataKey="time" stroke="#71767b" fontSize={11} tickLine={false} />
            <YAxis stroke="#71767b" fontSize={11} tickLine={false} domain={[0, 'auto']} />
            <Tooltip
              contentStyle={{
                background: '#16181c',
                border: '1px solid #2f3336',
                borderRadius: '8px',
              }}
              formatter={(value: number) => [`${value.toFixed(0)} ms`, 'TTFT']}
            />
            <Line
              type="monotone"
              dataKey="ttft_ms"
              stroke="#8b5cf6"
              strokeWidth={2}
              dot={false}
              name="Time to First Token"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}
