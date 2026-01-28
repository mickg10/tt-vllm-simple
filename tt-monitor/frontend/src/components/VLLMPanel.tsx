import { VLLMMetrics } from '../hooks/useDevices';

interface Props {
  metrics: VLLMMetrics;
}

export function VLLMPanel({ metrics }: Props) {
  if (!metrics.available) {
    return (
      <div className="vllm-panel offline">
        <h3>vLLM Server</h3>
        <div className="vllm-status">Offline</div>
      </div>
    );
  }

  const formatNumber = (n: number) => {
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
    if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
    return n.toFixed(0);
  };

  return (
    <div className="vllm-panel">
      <div className="vllm-header">
        <h3>vLLM Server</h3>
        <span className="model-name">{metrics.model_name}</span>
      </div>

      <div className="vllm-metrics">
        <div className="vllm-metric">
          <div className="label">Requests</div>
          <div className="value">
            <span className="running">{metrics.requests_running}</span>
            <span className="separator">/</span>
            <span className="waiting">{metrics.requests_waiting}</span>
          </div>
          <div className="sublabel">running / waiting</div>
        </div>

        <div className="vllm-metric">
          <div className="label">KV Cache</div>
          <div className="value">{metrics.gpu_cache_usage_percent.toFixed(1)}%</div>
        </div>

        <div className="vllm-metric">
          <div className="label">TTFT</div>
          <div className="value">{metrics.avg_time_to_first_token_ms.toFixed(0)}<span className="unit">ms</span></div>
        </div>

        <div className="vllm-metric">
          <div className="label">TPS</div>
          <div className="value">
            {metrics.avg_time_per_output_token_ms > 0
              ? (1000 / metrics.avg_time_per_output_token_ms).toFixed(1)
              : '0'}
            <span className="unit">tok/s</span>
          </div>
        </div>

        <div className="vllm-metric wide">
          <div className="label">Total Tokens</div>
          <div className="value">
            <span className="prompt">{formatNumber(metrics.prompt_tokens_total)}</span>
            <span className="separator">+</span>
            <span className="generated">{formatNumber(metrics.generation_tokens_total)}</span>
          </div>
          <div className="sublabel">prompt + generated</div>
        </div>
      </div>
    </div>
  );
}
