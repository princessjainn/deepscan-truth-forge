import { useState, useCallback } from "react";
import { Upload, Image, Video, Mic, FileWarning, X, Package } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ComplaintUploadProps {
  onAnalyze: () => void;
  isAnalyzing: boolean;
}

const ComplaintUpload = ({ onAnalyze, isAnalyzing }: ComplaintUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<{ name: string; type: string } | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    setUploadedFile({ name: "complaint_IMG_2847.jpg", type: "image" });
  }, []);

  const handleFileSelect = () => {
    setUploadedFile({ name: "complaint_IMG_2847.jpg", type: "image" });
  };

  const clearUpload = () => setUploadedFile(null);

  const orderContext = {
    orderId: "ORD-2024-847291",
    platform: "Zomato",
    items: ["Margherita Pizza", "Garlic Bread"],
  };

  return (
    <div className="forensic-card p-5 h-full flex flex-col">
      <h3 className="font-semibold mb-4">Complaint Media</h3>

      {/* Order Context - Compact */}
      <div className="mb-4 p-3 rounded-lg bg-muted/30 border border-border text-xs space-y-1">
        <div className="flex items-center gap-2 text-muted-foreground mb-2">
          <Package className="w-3.5 h-3.5" />
          <span className="font-medium uppercase tracking-wider">Order Context</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">ID:</span>
          <span className="font-mono">{orderContext.orderId}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">Platform:</span>
          <span className="text-primary font-medium">{orderContext.platform}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">Items:</span>
          <span className="text-right">{orderContext.items.join(", ")}</span>
        </div>
      </div>

      {!uploadedFile ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={handleFileSelect}
          className={`
            flex-1 border-2 border-dashed rounded-lg p-4 
            flex flex-col items-center justify-center gap-3 cursor-pointer
            transition-all duration-300 min-h-[160px]
            ${isDragging ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"}
          `}
        >
          <div className={`p-3 rounded-full bg-muted ${isDragging ? "bg-primary/20" : ""}`}>
            <Upload className={`w-5 h-5 ${isDragging ? "text-primary" : "text-muted-foreground"}`} />
          </div>
          
          <p className="text-sm font-medium">Upload Evidence</p>

          <div className="flex gap-2">
            {[Image, Video, Mic].map((Icon, i) => (
              <div key={i} className="p-1.5 rounded bg-muted">
                <Icon className="w-3.5 h-3.5 text-muted-foreground" />
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="flex-1 flex flex-col">
          <div className="flex-1 bg-muted/50 rounded-lg relative overflow-hidden group min-h-[160px] flex items-center justify-center">
            <div className="text-center">
              <div className="w-12 h-12 mx-auto mb-2 rounded-lg bg-forensic-navy flex items-center justify-center">
                <Image className="w-6 h-6 text-primary" />
              </div>
              <p className="font-mono text-xs">{uploadedFile.name}</p>
              <p className="text-xs text-muted-foreground">2.4 MB</p>
            </div>
            
            <button 
              onClick={clearUpload}
              className="absolute top-2 right-2 p-1.5 rounded-full bg-background/80 hover:bg-background opacity-0 group-hover:opacity-100 transition-opacity"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="mt-3 p-2.5 rounded-lg bg-warning/10 border border-warning/20 flex items-center gap-2">
            <FileWarning className="w-4 h-4 text-warning flex-shrink-0" />
            <p className="text-xs text-warning font-medium">Pending Verification</p>
          </div>
        </div>
      )}

      <Button 
        onClick={onAnalyze}
        disabled={!uploadedFile || isAnalyzing}
        className="w-full mt-4 h-10 font-semibold bg-gradient-to-r from-primary to-forensic-cyan hover:opacity-90"
      >
        {isAnalyzing ? (
          <span className="flex items-center gap-2">
            <div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
            Verifying...
          </span>
        ) : (
          "Run Verification"
        )}
      </Button>
    </div>
  );
};

export default ComplaintUpload;
