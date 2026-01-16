import { Eye, AlertTriangle, CheckCircle, XCircle, Scan } from "lucide-react";

interface VisualForensicsProps {
  isAnalyzed: boolean;
}

const VisualForensics = ({ isAnalyzed }: VisualForensicsProps) => {
  if (!isAnalyzed) {
    return (
      <div className="forensic-card p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-primary/10">
            <Eye className="w-5 h-5 text-primary" />
          </div>
          <h3 className="font-semibold">Visual Forensics</h3>
        </div>
        <div className="flex items-center justify-center h-40 text-muted-foreground text-sm">
          Run analysis to see results
        </div>
      </div>
    );
  }

  const visualChecks = [
    { check: "Face Swap Detection", status: "fail", confidence: 94, detail: "GAN-based replacement detected" },
    { check: "Boundary Artifacts", status: "fail", confidence: 87, detail: "Jaw and hairline inconsistencies" },
    { check: "Lighting Analysis", status: "warning", confidence: 68, detail: "Directional mismatch detected" },
    { check: "Skin Texture", status: "warning", confidence: 72, detail: "Synthetic smoothing patterns" },
    { check: "Eye Reflection", status: "pass", confidence: 91, detail: "Consistent light sources" },
  ];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "pass": return <CheckCircle className="w-4 h-4 text-success" />;
      case "fail": return <XCircle className="w-4 h-4 text-destructive" />;
      case "warning": return <AlertTriangle className="w-4 h-4 text-warning" />;
      default: return null;
    }
  };

  const getStatusBg = (status: string) => {
    switch (status) {
      case "pass": return "bg-success/10 border-success/20";
      case "fail": return "bg-destructive/10 border-destructive/20";
      case "warning": return "bg-warning/10 border-warning/20";
      default: return "";
    }
  };

  return (
    <div className="forensic-card p-6">
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-destructive/10">
            <Eye className="w-5 h-5 text-destructive" />
          </div>
          <h3 className="font-semibold">Visual Forensics</h3>
        </div>
        <span className="threat-high text-xs">
          <AlertTriangle className="w-3 h-3" />
          MANIPULATION
        </span>
      </div>

      <div className="space-y-3">
        {visualChecks.map((result, i) => (
          <div key={i} className={`p-3 rounded-lg border ${getStatusBg(result.status)}`}>
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2">
                {getStatusIcon(result.status)}
                <span className="text-sm font-medium">{result.check}</span>
              </div>
              <span className="text-xs font-mono text-muted-foreground">{result.confidence}%</span>
            </div>
            <p className="text-xs text-muted-foreground pl-6">{result.detail}</p>
          </div>
        ))}
      </div>

      <div className="mt-4 p-3 rounded-lg bg-destructive/10 border border-destructive/20">
        <div className="flex items-center gap-2">
          <Scan className="w-4 h-4 text-destructive" />
          <span className="text-sm font-medium text-destructive">Visual Manipulation Confirmed</span>
        </div>
      </div>
    </div>
  );
};

export default VisualForensics;
