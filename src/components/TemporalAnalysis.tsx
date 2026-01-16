import { Clock, AlertTriangle, Eye, Film } from "lucide-react";

interface TemporalAnalysisProps {
  isAnalyzed: boolean;
}

const TemporalAnalysis = ({ isAnalyzed }: TemporalAnalysisProps) => {
  if (!isAnalyzed) {
    return (
      <div className="forensic-card p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-primary/10">
            <Clock className="w-5 h-5 text-primary" />
          </div>
          <h3 className="font-semibold">Temporal Analysis</h3>
        </div>
        <div className="flex items-center justify-center h-40 text-muted-foreground text-sm">
          Run analysis to see results
        </div>
      </div>
    );
  }

  const temporalChecks = [
    { label: "Lip Sync Accuracy", value: 23, threshold: 45, status: "fail" },
    { label: "Blink Pattern", value: 3.2, threshold: 4.0, status: "fail", unit: "sec" },
    { label: "Frame Consistency", value: 68, threshold: 85, status: "warning" },
    { label: "Motion Blur", value: 74, threshold: 70, status: "pass" },
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case "fail": return "text-destructive";
      case "warning": return "text-warning";
      case "pass": return "text-success";
      default: return "text-muted-foreground";
    }
  };

  const getProgressColor = (status: string) => {
    switch (status) {
      case "fail": return "bg-destructive";
      case "warning": return "bg-warning";
      case "pass": return "bg-success";
      default: return "bg-muted";
    }
  };

  return (
    <div className="forensic-card p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-warning/10">
            <Clock className="w-5 h-5 text-warning" />
          </div>
          <h3 className="font-semibold">Temporal Analysis</h3>
        </div>
        <span className="threat-medium text-xs">
          <AlertTriangle className="w-3 h-3" />
          ANOMALIES
        </span>
      </div>

      <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 mb-4">
        <div className="flex items-center gap-2 text-destructive text-sm font-medium">
          <Eye className="w-4 h-4" />
          <span>Audio-Visual Desynchronization</span>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Lip movements lag behind audio by ~120ms (threshold: 45ms)
        </p>
      </div>

      <div className="space-y-4">
        {temporalChecks.map((check, i) => (
          <div key={i}>
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm">{check.label}</span>
              <span className={`text-xs font-mono ${getStatusColor(check.status)}`}>
                {check.value}{check.unit || "%"} / {check.threshold}{check.unit || "%"}
              </span>
            </div>
            <div className="h-1.5 bg-muted rounded-full overflow-hidden">
              <div 
                className={`h-full rounded-full ${getProgressColor(check.status)}`}
                style={{ width: `${Math.min((check.value / check.threshold) * 100, 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 p-3 rounded-lg bg-warning/10 border border-warning/20">
        <div className="flex items-center gap-2">
          <Film className="w-4 h-4 text-warning" />
          <span className="text-sm font-medium text-warning">Frame Interpolation Artifacts Detected</span>
        </div>
      </div>
    </div>
  );
};

export default TemporalAnalysis;
