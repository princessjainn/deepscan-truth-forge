import { Mic, AlertTriangle, User, Phone, Clock } from "lucide-react";

interface VoiceFraudDetectionProps {
  isAnalyzed: boolean;
}

const VoiceFraudDetection = ({ isAnalyzed }: VoiceFraudDetectionProps) => {
  if (!isAnalyzed) {
    return (
      <div className="forensic-card p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-primary/10">
            <Mic className="w-5 h-5 text-primary" />
          </div>
          <h3 className="font-semibold">Voice Fraud Detection</h3>
        </div>
        <div className="flex items-center justify-center h-40 text-muted-foreground text-sm">
          Run verification to see results
        </div>
      </div>
    );
  }

  const voiceMetrics = [
    { label: "Voice Authenticity", value: 34, status: "danger" },
    { label: "Natural Speech", value: 28, status: "danger" },
    { label: "Emotional Consistency", value: 62, status: "warning" },
    { label: "Background Authenticity", value: 89, status: "success" },
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
          <h3 className="font-semibold">Voice Fraud Detection</h3>
        </div>
        <span className="threat-high text-xs">
          <AlertTriangle className="w-3 h-3" />
          CLONE DETECTED
        </span>
      </div>

      <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 mb-4">
        <div className="flex items-center gap-2 text-destructive text-sm font-medium">
          <AlertTriangle className="w-4 h-4" />
          <span>Voice Cloning Detected</span>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          72% confidence synthetic voice
        </p>
      </div>

      <div className="space-y-3 mb-4">
        {voiceMetrics.map((metric, i) => (
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

      <div className="p-3 rounded-lg bg-muted/30 border border-border mb-4">
        <div className="flex items-center gap-2 mb-1">
          <User className="w-3.5 h-3.5 text-muted-foreground" />
          <span className="text-xs font-medium">Speaker Identity Reuse</span>
        </div>
        <p className="text-xs text-muted-foreground">
          Same voice in 4 previous calls
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 rounded-lg bg-muted/30 text-center">
          <Phone className="w-4 h-4 mx-auto mb-1 text-muted-foreground" />
          <div className="text-sm font-semibold">12</div>
          <div className="text-xs text-muted-foreground">Total Calls</div>
        </div>
        <div className="p-3 rounded-lg bg-muted/30 text-center">
          <Clock className="w-4 h-4 mx-auto mb-1 text-muted-foreground" />
          <div className="text-sm font-semibold">4m 32s</div>
          <div className="text-xs text-muted-foreground">Avg Duration</div>
        </div>
      </div>
    </div>
  );
};

export default VoiceFraudDetection;
