import { FileCode, AlertTriangle, CheckCircle, XCircle, Link2 } from "lucide-react";

interface MetadataAnalysisProps {
  isAnalyzed: boolean;
}

const MetadataAnalysis = ({ isAnalyzed }: MetadataAnalysisProps) => {
  if (!isAnalyzed) {
    return (
      <div className="forensic-card p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-primary/10">
            <FileCode className="w-5 h-5 text-primary" />
          </div>
          <h3 className="font-semibold">Metadata Analysis</h3>
        </div>
        <div className="flex items-center justify-center h-40 text-muted-foreground text-sm">
          Run analysis to see results
        </div>
      </div>
    );
  }

  const metadataChecks = [
    { field: "EXIF Integrity", value: "Partially Stripped", status: "warning" },
    { field: "Creation Date", value: "Jan 15, 2024 14:32", status: "pass" },
    { field: "Software", value: "Unknown / Modified", status: "fail" },
    { field: "Compression", value: "Multi-platform detected", status: "warning" },
    { field: "Hash Verification", value: "No original reference", status: "warning" },
  ];

  const compressionHistory = [
    { platform: "Instagram", detected: true },
    { platform: "WhatsApp", detected: true },
    { platform: "Telegram", detected: true },
    { platform: "Original", detected: false },
  ];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "pass": return <CheckCircle className="w-3.5 h-3.5 text-success" />;
      case "fail": return <XCircle className="w-3.5 h-3.5 text-destructive" />;
      case "warning": return <AlertTriangle className="w-3.5 h-3.5 text-warning" />;
      default: return null;
    }
  };

  return (
    <div className="forensic-card p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-warning/10">
            <FileCode className="w-5 h-5 text-warning" />
          </div>
          <h3 className="font-semibold">Metadata Analysis</h3>
        </div>
        <span className="threat-medium text-xs">
          <AlertTriangle className="w-3 h-3" />
          MODIFIED
        </span>
      </div>

      <div className="space-y-2 mb-4">
        {metadataChecks.map((check, i) => (
          <div key={i} className="flex items-center justify-between p-2 rounded bg-muted/30">
            <div className="flex items-center gap-2">
              {getStatusIcon(check.status)}
              <span className="text-sm">{check.field}</span>
            </div>
            <span className="text-xs font-mono text-muted-foreground">{check.value}</span>
          </div>
        ))}
      </div>

      <div className="p-3 rounded-lg bg-muted/30 border border-border">
        <div className="flex items-center gap-2 mb-2">
          <Link2 className="w-3.5 h-3.5 text-muted-foreground" />
          <span className="text-xs font-medium">Compression History</span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {compressionHistory.map((item, i) => (
            <span 
              key={i} 
              className={`px-2 py-0.5 rounded text-xs ${
                item.detected 
                  ? "bg-warning/20 text-warning" 
                  : "bg-muted text-muted-foreground"
              }`}
            >
              {item.platform}
            </span>
          ))}
        </div>
      </div>

      <div className="mt-4 p-3 rounded-lg bg-warning/10 border border-warning/20">
        <div className="flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-warning" />
          <span className="text-sm font-medium text-warning">Chain of Custody Compromised</span>
        </div>
      </div>
    </div>
  );
};

export default MetadataAnalysis;
