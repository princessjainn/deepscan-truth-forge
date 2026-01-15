import { Package, CheckCircle, XCircle, AlertTriangle, ArrowRight } from "lucide-react";

interface OrderConsistencyCheckProps {
  isAnalyzed: boolean;
}

const OrderConsistencyCheck = ({ isAnalyzed }: OrderConsistencyCheckProps) => {
  if (!isAnalyzed) {
    return (
      <div className="forensic-card p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-primary/10">
            <Package className="w-5 h-5 text-primary" />
          </div>
          <h3 className="font-semibold">Order-Media Consistency</h3>
        </div>
        <div className="flex items-center justify-center h-40 text-muted-foreground text-sm">
          Run verification to see results
        </div>
      </div>
    );
  }

  const consistencyResults = [
    { check: "Item Match", ordered: "Margherita Pizza", detected: "Margherita Pizza", status: "pass", confidence: 96 },
    { check: "Quantity", ordered: "1 Pizza, 1 Bread", detected: "1 Pizza visible", status: "warning", confidence: 72 },
    { check: "Packaging", ordered: "Zomato Delivery", detected: "Zomato branded", status: "pass", confidence: 89 },
    { check: "Timestamp", ordered: "Jan 15, 14:32", detected: "EXIF: Jan 15, 14:47", status: "pass", confidence: 100 },
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

  const overallScore = Math.round(consistencyResults.reduce((acc, r) => acc + r.confidence, 0) / consistencyResults.length);

  return (
    <div className="forensic-card p-6">
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-success/10">
            <Package className="w-5 h-5 text-success" />
          </div>
          <h3 className="font-semibold">Order-Media Consistency</h3>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold font-mono text-success">{overallScore}%</div>
          <div className="text-xs text-muted-foreground">Match Score</div>
        </div>
      </div>

      <div className="space-y-3">
        {consistencyResults.map((result, i) => (
          <div key={i} className={`p-3 rounded-lg border ${getStatusBg(result.status)}`}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                {getStatusIcon(result.status)}
                <span className="text-sm font-medium">{result.check}</span>
              </div>
              <span className="text-xs font-mono text-muted-foreground">{result.confidence}%</span>
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span className="bg-muted px-2 py-0.5 rounded truncate max-w-[120px]">{result.ordered}</span>
              <ArrowRight className="w-3 h-3 flex-shrink-0" />
              <span className="bg-muted px-2 py-0.5 rounded truncate max-w-[120px]">{result.detected}</span>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 p-3 rounded-lg bg-success/10 border border-success/20">
        <div className="flex items-center gap-2">
          <CheckCircle className="w-4 h-4 text-success" />
          <span className="text-sm font-medium text-success">Consistency Verified</span>
        </div>
      </div>
    </div>
  );
};

export default OrderConsistencyCheck;
