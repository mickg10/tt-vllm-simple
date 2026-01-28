interface Props {
  totals: {
    power: number;
    avg_temperature: number;
  };
  deviceCount: number;
}

export function OverviewPanel({ totals, deviceCount }: Props) {
  return (
    <div className="overview">
      <div className="overview-card">
        <div className="label">Total Devices</div>
        <div className="value">{deviceCount}</div>
      </div>

      <div className="overview-card">
        <div className="label">Total Power</div>
        <div className="value">
          {totals.power.toFixed(1)}
          <span className="unit">W</span>
        </div>
      </div>

      <div className="overview-card">
        <div className="label">Avg Temperature</div>
        <div className="value">
          {totals.avg_temperature.toFixed(1)}
          <span className="unit">C</span>
        </div>
      </div>
    </div>
  );
}
