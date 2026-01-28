import { Device } from '../hooks/useDevices';

interface Props {
  device: Device;
}

export function DeviceCard({ device }: Props) {
  const getTempColor = (temp: number) => {
    if (temp > 80) return '#f4212e';
    if (temp > 70) return '#ffa500';
    return '#00ba7c';
  };

  return (
    <div className="device-card">
      <div className="header">
        <div>
          <div className="device-name">{device.id}</div>
          <div className="device-type">
            {device.board_type.toUpperCase()} | PCI {device.pci_index}
          </div>
        </div>
        <span className={`status-badge ${device.status}`}>
          {device.status}
        </span>
      </div>

      <div className="metrics">
        <div className="metric">
          <div className="label">Temperature</div>
          <div className="value" style={{ color: getTempColor(device.temperature) }}>
            {device.temperature.toFixed(1)}
            <span className="unit">C</span>
          </div>
        </div>

        <div className="metric">
          <div className="label">Power</div>
          <div className="value">
            {device.power.toFixed(1)}
            <span className="unit">W</span>
          </div>
        </div>

        <div className="metric">
          <div className="label">AI Clock</div>
          <div className="value">
            {device.aiclk.toFixed(0)}
            <span className="unit">MHz</span>
          </div>
        </div>

        <div className="metric">
          <div className="label">Voltage</div>
          <div className="value">
            {device.voltage.toFixed(2)}
            <span className="unit">V</span>
          </div>
        </div>
      </div>

      <div className="device-footer">
        <span className="serial">{device.serial}</span>
        <span className="firmware">FW: {device.firmware}</span>
      </div>
    </div>
  );
}
