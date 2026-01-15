import { FileText, AlertCircle, Copy, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "@/hooks/use-toast";

const ForensicReport = ({ isAnalyzed }: { isAnalyzed: boolean }) => {
  const reportItems = [
    {
      severity: "critical",
      title: "Face Swap Technology Detected",
      description: "GAN-based face replacement identified with characteristic boundary artifacts around jaw and hairline regions. Confidence: 94%"
    },
    {
      severity: "critical",
      title: "Voice Cloning Identified",
      description: "Audio spectral analysis reveals synthetic voice generation patterns. Formant frequencies inconsistent with natural human speech production."
    },
    {
      severity: "critical",
      title: "Audio-Visual Desynchronization",
      description: "Lip movements lag behind audio by approximately 120ms, exceeding natural human tolerance threshold of 45ms."
    },
    {
      severity: "warning",
      title: "Abnormal Blink Rhythm",
      description: "Subject displays irregular blinking pattern with 3.2 second gaps, significantly exceeding natural 4-6 second average intervals."
    },
    {
      severity: "warning",
      title: "GAN Noise Artifacts",
      description: "Periodic noise patterns consistent with StyleGAN2 architecture detected in facial region. Anomaly score: 0.87"
    },
    {
      severity: "info",
      title: "Multiple Platform Compression",
      description: "File exhibits compression fingerprints from Instagram, WhatsApp, and Telegram, suggesting viral distribution across platforms."
    }
  ];

  const handleCopy = () => {
    const reportText = reportItems.map(item => `[${item.severity.toUpperCase()}] ${item.title}: ${item.description}`).join('\n\n');
    navigator.clipboard.writeText(reportText);
    toast({
      title: "Report Copied",
      description: "Forensic report has been copied to clipboard",
    });
  };

  if (!isAnalyzed) {
    return (
      <div className="forensic-card p-6">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 rounded-lg bg-primary/10">
            <FileText className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h2 className="font-semibold">Explainable Forensic Report</h2>
            <p className="text-xs text-muted-foreground">Human-readable analysis summary</p>
          </div>
        </div>
        
        <div className="flex items-center justify-center h-48 border border-dashed border-border rounded-lg">
          <p className="text-muted-foreground text-sm">Report will be generated after analysis</p>
        </div>
      </div>
    );
  }

  return (
    <div className="forensic-card p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-destructive/10">
            <FileText className="w-5 h-5 text-destructive" />
          </div>
          <div>
            <h2 className="font-semibold">Explainable Forensic Report</h2>
            <p className="text-xs text-muted-foreground">Human-readable analysis summary</p>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={handleCopy} className="gap-2">
            <Copy className="w-3.5 h-3.5" />
            Copy
          </Button>
          <Button variant="outline" size="sm" className="gap-2">
            <Download className="w-3.5 h-3.5" />
            Export
          </Button>
        </div>
      </div>

      <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2">
        {reportItems.map((item, index) => (
          <div
            key={index}
            className={`
              p-4 rounded-lg border-l-4 bg-muted/20 animate-fade-up
              ${item.severity === 'critical' 
                ? 'border-l-destructive' 
                : item.severity === 'warning'
                ? 'border-l-warning'
                : 'border-l-primary'
              }
            `}
            style={{ animationDelay: `${index * 100}ms` }}
          >
            <div className="flex items-start gap-3">
              <AlertCircle className={`
                w-5 h-5 flex-shrink-0 mt-0.5
                ${item.severity === 'critical' 
                  ? 'text-destructive' 
                  : item.severity === 'warning'
                  ? 'text-warning'
                  : 'text-primary'
                }
              `} />
              <div>
                <h4 className="font-medium text-sm mb-1">{item.title}</h4>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {item.description}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 p-4 bg-forensic-navy rounded-lg border border-border">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-2 h-2 rounded-full bg-destructive animate-pulse" />
          <span className="text-sm font-semibold">Final Assessment</span>
        </div>
        <p className="text-sm text-muted-foreground">
          This media exhibits <span className="text-destructive font-medium">multiple high-confidence manipulation indicators</span> consistent with 
          AI-generated deepfake content. The combination of face swap artifacts, synthetic voice patterns, and temporal inconsistencies 
          strongly suggests this is <span className="text-destructive font-medium">not authentic footage</span>.
        </p>
      </div>
    </div>
  );
};

export default ForensicReport;
