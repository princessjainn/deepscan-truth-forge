import { Mic, AlertTriangle, User, Phone, Clock } from "lucide-react";
import { Progress } from "@/components/ui/progress";

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
          <div>
            <h3 className="font-semibold">Voice Fraud Detection</h3>
            <p className="text-xs text-muted-foreground">Synthetic voice & clone detection</p>
          </div>
        </div>
        <div className="flex items-center justify-center h-48 text-muted-foreground text-sm">
          No voice data available...
        </div>
      </div>
    );
  }

  const voiceMetrics = [
    { label: "Voice Authenticity", value: 34, status: "danger" },
    { label: "Natural Speech Pattern", value: 28, status: "danger" },
    { label: "Emotional Consistency", value: 62, status: "warning" },
    { label: "Background Authenticity", value: 89, status: "success" }
  ];

  const getProgressColor = (status: string) => {
    switch (status) {
      case "danger":
        return "bg-destructive";
      case "warning":
        return "bg-warning";
      case "success":
        return "bg-success";
      default:
        return "bg-primary";
    }
  };

  return (
    <div className="forensic-card p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-destructive/10">
            <Mic className="w-5 h-5 text-destructive" />
          </div>
          <div>
            <h3 className="font-semibold">Voice Fraud Detection</h3>
            <p className="text-xs text-muted-foreground">Synthetic voice analysis</p>
          </div>
        </div>
        <span className="threat-high">
          <AlertTriangle className="w-3 h-3" />
          CLONE DETECTED
        </span>
      </div>

      {/* Alert Banner */}
      <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 mb-4">
        <div className="flex items-center gap-2 text-destructive text-sm font-medium">
          <AlertTriangle className="w-4 h-4" />
          <span>Possible Voice Cloning Detected</span>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Voice patterns indicate synthetic generation with 72% confidence
        </p>
      </div>

      {/* Voice Metrics */}
      <div className="space-y-4 mb-4">
        {voiceMetrics.map((metric, i) => (
          <div key={i}>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-sm">{metric.label}</span>
              <span className="text-xs font-mono text-muted-foreground">{metric.value}%</span>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div 
                className={`h-full rounded-full transition-all duration-500 ${getProgressColor(metric.status)}`}
                style={{ width: `${metric.value}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Speaker Identity */}
      <div className="p-3 rounded-lg bg-muted/30 border border-border mb-4">
        <div className="flex items-center gap-2 mb-2">
          <User className="w-4 h-4 text-muted-foreground" />
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Speaker Identity Tracking</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm">Identity Reuse Across Calls</span>
          <span className="px-2 py-0.5 rounded bg-warning/20 text-warning text-xs font-medium">HIGH</span>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Same voice signature detected in 4 previous complaint calls
        </p>
      </div>

      {/* Call Metadata */}
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
