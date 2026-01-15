import { TrendingUp, Shield, DollarSign, Clock, Ban, CheckCircle, ArrowUp, ArrowDown } from "lucide-react";

interface BusinessImpactDashboardProps {
  isAnalyzed: boolean;
}

const BusinessImpactDashboard = ({ isAnalyzed }: BusinessImpactDashboardProps) => {
  const metrics = [
    { icon: Shield, label: "Fraud Blocked", value: "12,847", change: "+23%", trend: "up", color: "text-success" },
    { icon: DollarSign, label: "Savings", value: "₹4.2Cr", change: "+18%", trend: "up", color: "text-primary" },
    { icon: Ban, label: "Refunds Prevented", value: "8,291", change: "+31%", trend: "up", color: "text-warning" },
    { icon: Clock, label: "Avg. Time", value: "1.2s", change: "-42%", trend: "down", color: "text-forensic-cyan" },
  ];

  const platformStats = [
    { name: "Zomato", blocked: 4821, savings: "₹1.8Cr", color: "bg-destructive" },
    { name: "Swiggy", blocked: 3927, savings: "₹1.4Cr", color: "bg-warning" },
    { name: "Blinkit", blocked: 2341, savings: "₹0.6Cr", color: "bg-success" },
    { name: "Zepto", blocked: 1758, savings: "₹0.4Cr", color: "bg-primary" },
  ];

  const maxBlocked = Math.max(...platformStats.map(p => p.blocked));

  return (
    <div className="forensic-card p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary/10">
            <TrendingUp className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h3 className="font-semibold">Business Impact</h3>
            <p className="text-xs text-muted-foreground">Last 30 days</p>
          </div>
        </div>
        <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-success/10 text-success text-xs font-medium">
          <CheckCircle className="w-3 h-3" />
          <span>Live</span>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
        {metrics.map((metric, i) => (
          <div key={i} className="p-3 rounded-lg bg-muted/30 border border-border">
            <div className="flex items-center justify-between mb-2">
              <metric.icon className={`w-4 h-4 ${metric.color}`} />
              <div className={`flex items-center gap-0.5 text-xs font-medium ${metric.trend === "up" ? "text-success" : "text-primary"}`}>
                {metric.trend === "up" ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />}
                {metric.change}
              </div>
            </div>
            <div className="text-xl font-bold font-mono">{metric.value}</div>
            <div className="text-xs text-muted-foreground">{metric.label}</div>
          </div>
        ))}
      </div>

      {/* Platform Breakdown */}
      <div className="mb-5">
        <h4 className="text-sm font-medium mb-3">Platform Breakdown</h4>
        <div className="space-y-2">
          {platformStats.map((platform, i) => (
            <div key={i} className="flex items-center gap-3">
              <div className="w-16 text-sm">{platform.name}</div>
              <div className="flex-1 h-5 bg-muted rounded overflow-hidden relative">
                <div 
                  className={`h-full ${platform.color} rounded`}
                  style={{ width: `${(platform.blocked / maxBlocked) * 100}%` }}
                />
                <span className="absolute inset-0 flex items-center px-2 text-xs font-mono">
                  {platform.blocked.toLocaleString()}
                </span>
              </div>
              <div className="w-16 text-right text-xs font-mono text-success">{platform.savings}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-3 gap-3 p-3 rounded-lg bg-gradient-to-r from-primary/10 to-forensic-cyan/10 border border-primary/20">
        <div className="text-center">
          <div className="text-xl font-bold font-mono gradient-text">99.2%</div>
          <div className="text-xs text-muted-foreground">Accuracy</div>
        </div>
        <div className="text-center border-x border-border">
          <div className="text-xl font-bold font-mono">2.4M</div>
          <div className="text-xs text-muted-foreground">Processed</div>
        </div>
        <div className="text-center">
          <div className="text-xl font-bold font-mono text-success">87%</div>
          <div className="text-xs text-muted-foreground">Auto-Approved</div>
        </div>
      </div>
    </div>
  );
};

export default BusinessImpactDashboard;
