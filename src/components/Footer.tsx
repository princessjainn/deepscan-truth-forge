import { ShieldCheck, Zap, Server, Globe } from "lucide-react";
import truefyLogo from "@/assets/truefy-logo.png";

const Footer = () => {
  const features = [
    { icon: ShieldCheck, label: "Privacy Compliant" },
    { icon: Zap, label: "Low-Latency API" },
    { icon: Server, label: "Enterprise Scale" },
    { icon: Globe, label: "Multi-Format Support" },
  ];

  return (
    <footer className="w-full border-t border-border bg-card/50 backdrop-blur-sm mt-8">
      <div className="container mx-auto px-6 py-6">
        <div className="flex flex-col lg:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <img src={truefyLogo} alt="TRUEFY" className="w-8 h-8 object-contain" />
            <div>
              <span className="text-sm font-semibold text-foreground">TRUEFY</span>
              <span className="text-sm text-muted-foreground"> — Deepfake Detection & Media Authenticity</span>
            </div>
          </div>

          <div className="flex items-center gap-4 flex-wrap justify-center">
            {features.map((f) => (
              <div key={f.label} className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <f.icon className="w-3.5 h-3.5 text-success" />
                <span>{f.label}</span>
              </div>
            ))}
          </div>
        </div>
        
        <div className="mt-4 pt-4 border-t border-border/50 text-center">
          <p className="text-xs text-muted-foreground">
            Designed for journalists, cybersecurity investigators, law enforcement, and content verification teams
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
