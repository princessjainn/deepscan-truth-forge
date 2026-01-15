import { Shield, Activity, Cpu } from "lucide-react";

const Header = () => {
  return (
    <header className="w-full border-b border-border bg-card/80 backdrop-blur-md sticky top-0 z-50">
      <div className="container mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="relative">
              <Shield className="w-10 h-10 text-primary" />
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-success rounded-full animate-pulse" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">
                Media Authenticity Forensic Dashboard
              </h1>
              <p className="text-sm text-muted-foreground">
                AI-powered deepfake detection & provenance analysis
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-6">
          <div className="hidden md:flex items-center gap-4 text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4" />
              <span>v2.4.1</span>
            </div>
          </div>
          
          <div className="status-active">
            <Cpu className="w-3.5 h-3.5" />
            <span>AI Engine Active</span>
            <span className="w-2 h-2 bg-success rounded-full animate-pulse" />
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
