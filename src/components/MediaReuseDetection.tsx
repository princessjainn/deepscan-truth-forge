import { Fingerprint, AlertTriangle, Clock, Link2, MapPin } from "lucide-react";

interface MediaReuseDetectionProps {
  isAnalyzed: boolean;
}

const MediaReuseDetection = ({ isAnalyzed }: MediaReuseDetectionProps) => {
  if (!isAnalyzed) {
    return (
      <div className="forensic-card p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-primary/10">
            <Fingerprint className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h3 className="font-semibold">Media Reuse Detection</h3>
            <p className="text-xs text-muted-foreground">Fingerprint matching across claims</p>
          </div>
        </div>
        <div className="flex items-center justify-center h-48 text-muted-foreground text-sm">
          Awaiting media upload...
        </div>
      </div>
    );
  }

  const reuseMatches = [
    {
      claimId: "CLM-2024-738291",
      platform: "Swiggy",
      date: "Jan 12, 2024",
      similarity: 97,
      location: "Mumbai"
    },
    {
      claimId: "CLM-2024-692847",
      platform: "Zomato",
      date: "Jan 8, 2024",
      similarity: 94,
      location: "Mumbai"
    },
    {
      claimId: "CLM-2024-584729",
      platform: "Blinkit",
      date: "Dec 28, 2023",
      similarity: 89,
      location: "Pune"
    }
  ];

  return (
    <div className="forensic-card p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-warning/10">
            <Fingerprint className="w-5 h-5 text-warning" />
          </div>
          <div>
            <h3 className="font-semibold">Media Reuse Detection</h3>
            <p className="text-xs text-muted-foreground">Cross-platform fingerprint matching</p>
          </div>
        </div>
        <span className="threat-high">
          <AlertTriangle className="w-3 h-3" />
          {reuseMatches.length} MATCHES
        </span>
      </div>

      <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 mb-4">
        <div className="flex items-center gap-2 text-destructive text-sm font-medium">
          <AlertTriangle className="w-4 h-4" />
          <span>Similar media detected in {reuseMatches.length} previous complaints</span>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Evidence suggests potential fraud pattern across multiple refund claims
        </p>
      </div>

      <div className="space-y-3">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Matched Claims</h4>
        
        {reuseMatches.map((match, i) => (
          <div key={i} className="p-3 rounded-lg bg-muted/30 border border-border hover:border-primary/30 transition-colors">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Link2 className="w-3.5 h-3.5 text-primary" />
                <span className="font-mono text-sm text-primary">{match.claimId}</span>
              </div>
              <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-destructive/20 text-destructive text-xs font-medium">
                {match.similarity}% Match
              </div>
            </div>
            <div className="flex items-center gap-4 text-xs text-muted-foreground">
              <div className="flex items-center gap-1">
                <span className="font-medium text-foreground">{match.platform}</span>
              </div>
              <div className="flex items-center gap-1">
                <Clock className="w-3 h-3" />
                <span>{match.date}</span>
              </div>
              <div className="flex items-center gap-1">
                <MapPin className="w-3 h-3" />
                <span>{match.location}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 p-3 rounded-lg bg-muted/50 text-center">
        <p className="text-xs text-muted-foreground">
          <span className="font-semibold text-foreground">Fraud Pattern Score:</span>{" "}
          <span className="text-destructive font-mono">HIGH RISK</span>
        </p>
      </div>
    </div>
  );
};

export default MediaReuseDetection;
