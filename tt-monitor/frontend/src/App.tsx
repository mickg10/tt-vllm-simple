import { useState } from 'react';
import { useDevices, useVLLM, useDeviceHistory, useVLLMHistory, TimeRange } from './hooks/useDevices';
import { DeviceCard } from './components/DeviceCard';
import { OverviewPanel } from './components/OverviewPanel';
import { HistoryChart } from './components/HistoryChart';
import { VLLMPanel } from './components/VLLMPanel';
import { VLLMChart } from './components/VLLMChart';

const TIME_RANGE_OPTIONS: { value: TimeRange; label: string }[] = [
  { value: '10m', label: '10 Minutes' },
  { value: '1h', label: '1 Hour' },
  { value: '6h', label: '6 Hours' },
  { value: '24h', label: '24 Hours' },
  { value: '1w', label: '1 Week' },
];

function App() {
  const [timeRange, setTimeRange] = useState<TimeRange>('1h');
  const { data, loading, error } = useDevices(5000);
  const { data: vllmData } = useVLLM(5000);
  const { data: deviceHistory } = useDeviceHistory(timeRange, 10000);
  const { data: vllmHistory } = useVLLMHistory(timeRange, 10000);

  return (
    <div className="app">
      <header className="header">
        <h1>TT-Monitor</h1>
        <div className="header-right">
          <select
            className="time-range-select"
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value as TimeRange)}
          >
            {TIME_RANGE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          <div className="status">
            <span className={`status-dot ${error ? 'error' : ''}`}></span>
            <span>
              {loading
                ? 'Loading...'
                : error
                ? 'Connection Error'
                : `${data?.device_count || 0} devices online`}
            </span>
          </div>
        </div>
      </header>

      {error && (
        <div className="error-message">
          Error: {error}. Make sure the backend is running and devices are accessible.
        </div>
      )}

      {data && (
        <>
          <div className="overview-row">
            <OverviewPanel totals={data.totals} deviceCount={data.device_count} />
            {vllmData && <VLLMPanel metrics={vllmData} />}
          </div>

          <div className="devices-grid">
            {data.devices.map((device) => (
              <DeviceCard key={device.id} device={device} />
            ))}
          </div>

          <div className="charts-section">
            <h2>Device Performance History ({TIME_RANGE_OPTIONS.find(o => o.value === timeRange)?.label})</h2>
            {deviceHistory && Object.keys(deviceHistory.devices).length > 0 && (
              <HistoryChart history={deviceHistory} />
            )}
          </div>

          {vllmHistory && vllmHistory.history.length > 0 && (
            <div className="charts-section">
              <h2>vLLM Performance History ({TIME_RANGE_OPTIONS.find(o => o.value === timeRange)?.label})</h2>
              <VLLMChart history={vllmHistory} />
            </div>
          )}
        </>
      )}

      {!loading && !error && (!data || data.devices.length === 0) && (
        <div className="no-data">No Tenstorrent devices found</div>
      )}
    </div>
  );
}

export default App;
