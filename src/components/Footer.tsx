import { Newspaper, Search, Users } from "lucide-react";
import truefyLogo from "@/assets/truefy-logo.png";

const Footer = () => {
  const useCases = [
    { icon: Newspaper, label: "Journalism" },
    { icon: Search, label: "Cybercrime" },
    { icon: Users, label: "Social Media Moderation" },
  ];

  return (
    <footer className="w-full border-t border-border bg-card/50 backdrop-blur-sm mt-8">
      <div className="container mx-auto px-6 py-6">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <img src={truefyLogo} alt="TRUEFY" className="w-6 h-6 object-contain" />
            <span className="text-sm text-muted-foreground">
              <span className="font-semibold text-foreground">TRUEFY</span> — Designed for real-world media verification
            </span>
          </div>

          <div className="flex items-center gap-6">
            <span className="text-xs text-muted-foreground uppercase tracking-wider">Use Cases:</span>
            {useCases.map((item) => (
              <div key={item.label} className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors">
                <item.icon className="w-4 h-4" />
                <span>{item.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
