import { AlertTriangle, ShieldCheck, ShieldX, TrendingUp, Eye, AlertCircle, CheckCircle } from "lucide-react";

interface FraudRiskScoreProps {
  isAnalyzed: boolean;
  isAnalyzing: boolean;
}

const FraudRiskScore = ({ isAnalyzed, isAnalyzing }: FraudRiskScoreProps) => {
  if (!isAnalyzed && !isAnalyzing) {
    return (
      <div className="forensic-card p-8 flex flex-col items-center justify-center h-full min-h-[360px]">
        <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
          <ShieldCheck className="w-8 h-8 text-muted-foreground" />
        </div>
        <h3 className="text-lg font-semibold mb-2">Awaiting Verification</h3>
        <p className="text-muted-foreground text-center text-sm max-w-xs">
          Submit complaint evidence to receive fraud risk assessment
        </p>
      </div>
    );
  }

  if (isAnalyzing) {
    return (
      <div className="forensic-card-glow p-8 flex flex-col items-center justify-center h-full min-h-[360px] relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent" />
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute inset-x-0 h-px bg-gradient-to-r from-transparent via-primary to-transparent scan-line" />
        </div>
        
        <div className="relative z-10 flex flex-col items-center">
          <div className="w-20 h-20 rounded-full border-4 border-primary/30 flex items-center justify-center mb-4">
            <div className="w-16 h-16 rounded-full border-4 border-primary border-t-transparent animate-spin" />
          </div>
          <h3 className="text-lg font-semibold mb-2">Analyzing Claim</h3>
          
          <div className="flex flex-wrap gap-2 justify-center mt-4">
            {["AI Detection", "Fingerprint", "Order Match", "Voice"].map((step, i) => (
              <div 
                key={step}
                className="px-2.5 py-1 rounded-full text-xs font-medium bg-muted animate-pulse"
                style={{ animationDelay: `${i * 200}ms` }}
              >
                {step}
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="forensic-card-danger p-6 relative overflow-hidden h-full">
      <div className="absolute inset-0 bg-gradient-to-br from-destructive/10 via-transparent to-transparent" />
      
      <div className="relative z-10">
        {/* Header */}
        <div className="flex items-start gap-4 mb-5">
          <div className="p-2.5 rounded-xl bg-destructive/20">
            <ShieldX className="w-7 h-7 text-destructive" />
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1 flex-wrap">
              <h2 className="text-xl font-bold text-destructive">HIGH FRAUD RISK</h2>
              <span className="threat-high text-xs">
                <AlertTriangle className="w-3 h-3" />
                MANUAL REVIEW
              </span>
            </div>
            <p className="text-sm text-muted-foreground">
              Multiple fraud indicators detected
            </p>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-3 mb-5">
          <div className="bg-background/50 rounded-lg p-3 text-center">
            <div className="text-2xl font-bold text-destructive font-mono">76%</div>
            <div className="text-xs text-muted-foreground">Fraud Risk</div>
          </div>
          <div className="bg-background/50 rounded-lg p-3 text-center">
            <div className="text-2xl font-bold text-warning font-mono">84%</div>
            <div className="text-xs text-muted-foreground">AI-Generated</div>
          </div>
          <div className="bg-background/50 rounded-lg p-3 text-center">
            <div className="flex items-center justify-center gap-1">
              <TrendingUp className="w-4 h-4 text-success" />
              <span className="text-lg font-bold text-success">HIGH</span>
            </div>
            <div className="text-xs text-muted-foreground">Confidence</div>
          </div>
        </div>

        {/* Action */}
        <div className="p-3 rounded-lg bg-warning/10 border border-warning/30 mb-4">
          <div className="flex items-center gap-2 mb-1">
            <Eye className="w-4 h-4 text-warning" />
            <span className="text-sm font-semibold text-warning">Recommendation</span>
          </div>
          <p className="text-xs text-muted-foreground">
            Route to human reviewer. Do not auto-approve. Est. savings: ₹847
          </p>
        </div>

        {/* Detection Summary */}
        <div className="space-y-2">
          <div className="flex items-center justify-between p-2.5 rounded-lg bg-destructive/10 border border-destructive/20">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-3.5 h-3.5 text-destructive" />
              <span className="text-sm">AI-Generated Image</span>
            </div>
            <span className="text-xs font-mono text-destructive">84%</span>
          </div>
          
          <div className="flex items-center justify-between p-2.5 rounded-lg bg-warning/10 border border-warning/20">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-3.5 h-3.5 text-warning" />
              <span className="text-sm">3 Previous Claims Match</span>
            </div>
            <span className="text-xs font-mono text-warning">Match</span>
          </div>
          
          <div className="flex items-center justify-between p-2.5 rounded-lg bg-success/10 border border-success/20">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-3.5 h-3.5 text-success" />
              <span className="text-sm">Order-Media Consistent</span>
            </div>
            <span className="text-xs font-mono text-success">OK</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FraudRiskScore;
