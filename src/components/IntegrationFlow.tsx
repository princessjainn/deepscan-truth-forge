import { ArrowRight, Upload, Cpu, Shield, CheckCircle, XCircle, Eye } from "lucide-react";

interface IntegrationFlowProps {
  isAnalyzed: boolean;
  isAnalyzing: boolean;
}

const IntegrationFlow = ({ isAnalyzed, isAnalyzing }: IntegrationFlowProps) => {
  const steps = [
    {
      icon: Upload,
      label: "Complaint Submitted",
      description: "User uploads evidence",
      status: isAnalyzed || isAnalyzing ? "complete" : "pending"
    },
    {
      icon: Cpu,
      label: "TRUEFY API",
      description: "Fraud detection pipeline",
      status: isAnalyzing ? "active" : isAnalyzed ? "complete" : "pending"
    },
    {
      icon: Shield,
      label: "Verification",
      description: "Multi-modal analysis",
      status: isAnalyzing ? "active" : isAnalyzed ? "complete" : "pending"
    },
    {
      icon: isAnalyzed ? Eye : CheckCircle,
      label: isAnalyzed ? "Manual Review" : "Decision",
      description: isAnalyzed ? "Flagged for review" : "Action recommendation",
      status: isAnalyzed ? "flagged" : "pending"
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case "complete":
        return "bg-success text-success-foreground border-success";
      case "active":
        return "bg-primary text-primary-foreground border-primary animate-pulse";
      case "flagged":
        return "bg-warning text-warning-foreground border-warning";
      default:
        return "bg-muted text-muted-foreground border-border";
    }
  };

  const getLineColor = (status: string) => {
    switch (status) {
      case "complete":
        return "bg-success";
      case "active":
        return "bg-primary animate-pulse";
      default:
        return "bg-border";
    }
  };

  return (
    <div className="forensic-card p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-lg bg-primary/10">
          <ArrowRight className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h3 className="font-semibold">TRUEFY Integration Flow</h3>
          <p className="text-xs text-muted-foreground">Real-time complaint verification pipeline</p>
        </div>
      </div>

      <div className="flex items-center justify-between">
        {steps.map((step, i) => (
          <div key={i} className="flex items-center flex-1">
            {/* Step Circle */}
            <div className="flex flex-col items-center">
              <div className={`w-12 h-12 rounded-full border-2 flex items-center justify-center ${getStatusColor(step.status)}`}>
                <step.icon className="w-5 h-5" />
              </div>
              <div className="mt-2 text-center">
                <div className="text-xs font-semibold">{step.label}</div>
                <div className="text-xs text-muted-foreground">{step.description}</div>
              </div>
            </div>
            
            {/* Connector Line */}
            {i < steps.length - 1 && (
              <div className={`flex-1 h-0.5 mx-2 ${getLineColor(step.status)}`} />
            )}
          </div>
        ))}
      </div>

      {/* Power Statement */}
      <div className="mt-6 p-4 rounded-lg bg-gradient-to-r from-primary/5 to-forensic-cyan/5 border border-primary/20 text-center">
        <p className="text-sm font-medium text-foreground">
          "TRUEFY prevents AI-driven refund fraud by verifying complaint media authenticity in real time for delivery platforms."
        </p>
      </div>
    </div>
  );
};

export default IntegrationFlow;
