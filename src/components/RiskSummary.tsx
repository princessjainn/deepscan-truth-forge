import { AlertTriangle, ShieldCheck, ShieldX, TrendingUp, Gauge } from "lucide-react";

interface RiskSummaryProps {
  isAnalyzed: boolean;
  isAnalyzing: boolean;
}

const RiskSummary = ({ isAnalyzed, isAnalyzing }: RiskSummaryProps) => {
  if (!isAnalyzed && !isAnalyzing) {
    return (
      <div className="forensic-card p-8 flex flex-col items-center justify-center h-full min-h-[300px]">
        <div className="w-20 h-20 rounded-full bg-muted flex items-center justify-center mb-6">
          <ShieldCheck className="w-10 h-10 text-muted-foreground" />
        </div>
        <h3 className="text-xl font-semibold mb-2">Awaiting Analysis</h3>
        <p className="text-muted-foreground text-center max-w-sm">
          Upload a media file and run forensic analysis to see authenticity results
        </p>
      </div>
    );
  }

  if (isAnalyzing) {
    return (
      <div className="forensic-card-glow p-8 flex flex-col items-center justify-center h-full min-h-[300px] relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent" />
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute inset-x-0 h-px bg-gradient-to-r from-transparent via-primary to-transparent scan-line" />
        </div>
        
        <div className="relative z-10 flex flex-col items-center">
          <div className="relative mb-6">
            <div className="w-24 h-24 rounded-full border-4 border-primary/30 flex items-center justify-center">
              <div className="w-20 h-20 rounded-full border-4 border-primary border-t-transparent animate-spin" />
            </div>
          </div>
          <h3 className="text-xl font-semibold mb-2">Analyzing Media</h3>
          <p className="text-muted-foreground text-center text-sm">
            Running multi-modal forensic detection...
          </p>
          
          <div className="mt-6 flex flex-wrap gap-2 justify-center">
            {["Face Detection", "GAN Artifacts", "Audio Sync", "Metadata"].map((step, i) => (
              <div 
                key={step}
                className="px-3 py-1.5 rounded-full text-xs font-medium bg-muted animate-pulse"
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
    <div className="forensic-card-danger p-6 relative overflow-hidden">
      {/* Background gradient effect */}
      <div className="absolute inset-0 bg-gradient-to-br from-destructive/10 via-transparent to-transparent" />
      
      <div className="relative z-10">
        {/* Main verdict */}
        <div className="flex items-start gap-4 mb-6">
          <div className="p-3 rounded-xl bg-destructive/20">
            <ShieldX className="w-8 h-8 text-destructive" />
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-1">
              <h2 className="text-2xl font-bold text-destructive">LIKELY MANIPULATED</h2>
              <span className="threat-high">
                <AlertTriangle className="w-3 h-3" />
                HIGH RISK
              </span>
            </div>
            <p className="text-sm text-muted-foreground">
              Multiple manipulation indicators detected with high confidence
            </p>
          </div>
        </div>

        {/* Stats grid */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="bg-background/50 rounded-lg p-4 text-center">
            <div className="text-3xl font-bold text-destructive font-mono mb-1">92%</div>
            <div className="text-xs text-muted-foreground uppercase tracking-wider">Fake Probability</div>
          </div>
          <div className="bg-background/50 rounded-lg p-4 text-center">
            <div className="flex items-center justify-center gap-1 mb-1">
              <TrendingUp className="w-5 h-5 text-destructive" />
              <span className="text-2xl font-bold text-destructive">HIGH</span>
            </div>
            <div className="text-xs text-muted-foreground uppercase tracking-wider">Reliability Score</div>
          </div>
          <div className="bg-background/50 rounded-lg p-4 text-center">
            <div className="text-3xl font-bold text-warning font-mono mb-1">3</div>
            <div className="text-xs text-muted-foreground uppercase tracking-wider">Manipulation Types</div>
          </div>
        </div>

        {/* Threat level breakdown */}
        <div className="space-y-3">
          <h4 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">Threat Assessment</h4>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between p-3 rounded-lg bg-destructive/10 border border-destructive/20">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 rounded-full bg-destructive" />
                <span className="text-sm font-medium">Identity Impersonation + Voice Cloning</span>
              </div>
              <span className="threat-high">Critical</span>
            </div>
            
            <div className="flex items-center justify-between p-3 rounded-lg bg-warning/10 border border-warning/20">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 rounded-full bg-warning" />
                <span className="text-sm font-medium">Face Manipulation Detected</span>
              </div>
              <span className="threat-medium">Medium</span>
            </div>
            
            <div className="flex items-center justify-between p-3 rounded-lg bg-success/10 border border-success/20">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 rounded-full bg-success" />
                <span className="text-sm font-medium">Standard Filters Applied</span>
              </div>
              <span className="threat-low">Low</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RiskSummary;
