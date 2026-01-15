import { Building2, ShieldCheck, Zap, Server } from "lucide-react";
import truefyLogo from "@/assets/truefy-logo.png";

const Footer = () => {
  const platforms = ["Zomato", "Swiggy", "Blinkit", "Zepto"];
  
  const features = [
    { icon: ShieldCheck, label: "GDPR Compliant" },
    { icon: Zap, label: "Low-Latency API" },
    { icon: Server, label: "Enterprise Scale" },
  ];

  return (
    <footer className="w-full border-t border-border bg-card/50 backdrop-blur-sm mt-8">
      <div className="container mx-auto px-6 py-6">
        <div className="flex flex-col lg:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <img src={truefyLogo} alt="TRUEFY" className="w-8 h-8 object-contain" />
            <div>
              <span className="text-sm font-semibold text-foreground">TRUEFY</span>
              <span className="text-sm text-muted-foreground"> — Enterprise Media Trust & Fraud Intelligence</span>
            </div>
          </div>

          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <Building2 className="w-4 h-4 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Integrated with:</span>
              {platforms.map((p, i) => (
                <span key={p} className="text-xs text-primary font-medium">
                  {p}{i < platforms.length - 1 && <span className="text-muted-foreground mx-1">•</span>}
                </span>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-4">
            {features.map((f) => (
              <div key={f.label} className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <f.icon className="w-3.5 h-3.5 text-success" />
                <span>{f.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
