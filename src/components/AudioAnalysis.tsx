import { Mic, AlertTriangle, Activity, AudioWaveform } from "lucide-react";

interface AudioAnalysisProps {
  isAnalyzed: boolean;
}

const AudioAnalysis = ({ isAnalyzed }: AudioAnalysisProps) => {
  if (!isAnalyzed) {
    return (
      <div className="forensic-card p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-primary/10">
            <Mic className="w-5 h-5 text-primary" />
          </div>
          <h3 className="font-semibold">Audio Analysis</h3>
        </div>
        <div className="flex items-center justify-center h-40 text-muted-foreground text-sm">
          Run analysis to see results
        </div>
      </div>
    );
  }

  const audioMetrics = [
    { label: "Voice Authenticity", value: 34, status: "danger" },
    { label: "Formant Analysis", value: 28, status: "danger" },
    { label: "Speech Cadence", value: 45, status: "warning" },
    { label: "Background Audio", value: 89, status: "success" },
  ];

  const getProgressColor = (status: string) => {
    switch (status) {
      case "danger": return "bg-destructive";
      case "warning": return "bg-warning";
      case "success": return "bg-success";
      default: return "bg-primary";
    }
  };

  return (
    <div className="forensic-card p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-destructive/10">
            <Mic className="w-5 h-5 text-destructive" />
          </div>
          <h3 className="font-semibold">Audio Analysis</h3>
        </div>
        <span className="threat-high text-xs">
          <AlertTriangle className="w-3 h-3" />
          SYNTHETIC VOICE
        </span>
      </div>

      <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 mb-4">
        <div className="flex items-center gap-2 text-destructive text-sm font-medium">
          <AlertTriangle className="w-4 h-4" />
          <span>Voice Cloning Detected</span>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Neural TTS patterns identified in audio stream
        </p>
      </div>

      <div className="space-y-3 mb-4">
        {audioMetrics.map((metric, i) => (
          <div key={i}>
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm">{metric.label}</span>
              <span className="text-xs font-mono text-muted-foreground">{metric.value}%</span>
            </div>
            <div className="h-1.5 bg-muted rounded-full overflow-hidden">
              <div 
                className={`h-full rounded-full ${getProgressColor(metric.status)}`}
                style={{ width: `${metric.value}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 rounded-lg bg-muted/30 text-center">
          <AudioWaveform className="w-4 h-4 mx-auto mb-1 text-muted-foreground" />
          <div className="text-sm font-semibold">Spectral</div>
          <div className="text-xs text-muted-foreground">Anomalies Found</div>
        </div>
        <div className="p-3 rounded-lg bg-muted/30 text-center">
          <Activity className="w-4 h-4 mx-auto mb-1 text-muted-foreground" />
          <div className="text-sm font-semibold">87%</div>
          <div className="text-xs text-muted-foreground">Synthetic Prob.</div>
        </div>
      </div>
    </div>
  );
};

export default AudioAnalysis;
