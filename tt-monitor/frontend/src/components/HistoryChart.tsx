import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { DeviceHistoryResponse } from '../hooks/useDevices';

interface Props {
  history: DeviceHistoryResponse;
}

const COLORS = ['#1d9bf0', '#00ba7c', '#f91880', '#ffa500', '#8b5cf6', '#ef4444'];

export function HistoryChart({ history }: Props) {
  const deviceIds = Object.keys(history.devices);
  if (deviceIds.length === 0) {
    return <div className="no-data">No history data available</div>;
  }

  // Get all unique timestamps and create merged data
  const allTimestamps = new Set<number>();
  deviceIds.forEach((id) => {
    history.devices[id].forEach((point) => {
      allTimestamps.add(point.timestamp);
    });
  });

  const sortedTimestamps = Array.from(allTimestamps).sort((a, b) => a - b);

  const chartData = sortedTimestamps.map((ts) => {
    const point: Record<string, number | string> = {
      time: new Date(ts * 1000).toLocaleTimeString(),
      timestamp: ts,
    };

    deviceIds.forEach((id) => {
      const devicePoint = history.devices[id].find((p) => p.timestamp === ts);
      if (devicePoint) {
        point[`${id}_temp`] = devicePoint.temperature || 0;
        point[`${id}_power`] = devicePoint.power || 0;
      }
    });

    return point;
  });

  return (
    <>
      <div className="chart-container">
        <h3 style={{ fontSize: '14px', color: '#71767b', marginBottom: '12px' }}>
          Temperature (C)
        </h3>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2f3336" />
            <XAxis
              dataKey="time"
              stroke="#71767b"
              fontSize={11}
              tickLine={false}
            />
            <YAxis
              stroke="#71767b"
              fontSize={11}
              tickLine={false}
              domain={['auto', 'auto']}
            />
            <Tooltip
              contentStyle={{
                background: '#16181c',
                border: '1px solid #2f3336',
                borderRadius: '8px',
              }}
            />
            <Legend />
            {deviceIds.map((id, idx) => (
              <Line
                key={id}
                type="monotone"
                dataKey={`${id}_temp`}
                stroke={COLORS[idx % COLORS.length]}
                strokeWidth={2}
                dot={false}
                name={id}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-container">
        <h3 style={{ fontSize: '14px', color: '#71767b', marginBottom: '12px' }}>
          Power (W)
        </h3>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2f3336" />
            <XAxis
              dataKey="time"
              stroke="#71767b"
              fontSize={11}
              tickLine={false}
            />
            <YAxis
              stroke="#71767b"
              fontSize={11}
              tickLine={false}
              domain={['auto', 'auto']}
            />
            <Tooltip
              contentStyle={{
                background: '#16181c',
                border: '1px solid #2f3336',
                borderRadius: '8px',
              }}
            />
            <Legend />
            {deviceIds.map((id, idx) => (
              <Line
                key={id}
                type="monotone"
                dataKey={`${id}_power`}
                stroke={COLORS[idx % COLORS.length]}
                strokeWidth={2}
                dot={false}
                name={id}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}
