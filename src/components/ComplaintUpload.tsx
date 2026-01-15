import { useState, useCallback } from "react";
import { Upload, Image, Video, Mic, FileWarning, X, Package, ClipboardList } from "lucide-react";
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
    setUploadedFile({
      name: "complaint_evidence_IMG_2847.jpg",
      type: "image"
    });
  }, []);

  const handleFileSelect = () => {
    setUploadedFile({
      name: "complaint_evidence_IMG_2847.jpg",
      type: "image"
    });
  };

  const clearUpload = () => {
    setUploadedFile(null);
  };

  // Simulated order context
  const orderContext = {
    orderId: "ORD-2024-847291",
    platform: "Zomato",
    items: ["Margherita Pizza", "Garlic Bread"],
    timestamp: "2024-01-15 14:32:18"
  };

  return (
    <div className="forensic-card p-6 h-full flex flex-col">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-lg bg-primary/10">
          <ClipboardList className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h2 className="font-semibold">Complaint Media Intake</h2>
          <p className="text-xs text-muted-foreground">Upload refund claim evidence</p>
        </div>
      </div>

      {/* Order Context */}
      <div className="mb-4 p-3 rounded-lg bg-muted/30 border border-border">
        <div className="flex items-center gap-2 mb-2">
          <Package className="w-4 h-4 text-muted-foreground" />
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Order Context</span>
        </div>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div>
            <span className="text-muted-foreground">Order ID:</span>
            <span className="ml-2 font-mono text-foreground">{orderContext.orderId}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Platform:</span>
            <span className="ml-2 font-medium text-primary">{orderContext.platform}</span>
          </div>
          <div className="col-span-2">
            <span className="text-muted-foreground">Items:</span>
            <span className="ml-2 text-foreground">{orderContext.items.join(", ")}</span>
          </div>
        </div>
      </div>

      {!uploadedFile ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={handleFileSelect}
          className={`
            flex-1 border-2 border-dashed rounded-lg p-6 
            flex flex-col items-center justify-center gap-3 cursor-pointer
            transition-all duration-300 min-h-[200px]
            ${isDragging 
              ? "border-primary bg-primary/5" 
              : "border-border hover:border-primary/50 hover:bg-muted/30"
            }
          `}
        >
          <div className={`
            p-3 rounded-full bg-muted transition-all duration-300
            ${isDragging ? "scale-110 bg-primary/20" : ""}
          `}>
            <Upload className={`w-6 h-6 ${isDragging ? "text-primary" : "text-muted-foreground"}`} />
          </div>
          
          <div className="text-center">
            <p className="font-medium mb-1 text-sm">
              {isDragging ? "Drop evidence here" : "Upload Complaint Evidence"}
            </p>
            <p className="text-xs text-muted-foreground">
              Drag & drop or click to browse
            </p>
          </div>

          <div className="flex flex-wrap justify-center gap-2 mt-2">
            <div className="flex items-center gap-1 text-xs text-muted-foreground bg-muted px-2 py-1 rounded-full">
              <Image className="w-3 h-3" />
              <span>Photos</span>
            </div>
            <div className="flex items-center gap-1 text-xs text-muted-foreground bg-muted px-2 py-1 rounded-full">
              <Video className="w-3 h-3" />
              <span>Videos</span>
            </div>
            <div className="flex items-center gap-1 text-xs text-muted-foreground bg-muted px-2 py-1 rounded-full">
              <Mic className="w-3 h-3" />
              <span>Voice</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="flex-1 flex flex-col">
          <div className="flex-1 bg-muted/50 rounded-lg relative overflow-hidden group min-h-[200px]">
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="w-16 h-16 mx-auto mb-3 rounded-xl bg-forensic-navy flex items-center justify-center">
                  <Image className="w-8 h-8 text-primary" />
                </div>
                <p className="font-mono text-sm">{uploadedFile.name}</p>
                <p className="text-xs text-muted-foreground mt-1">Image • 2.4 MB</p>
              </div>
            </div>
            
            <button 
              onClick={clearUpload}
              className="absolute top-3 right-3 p-2 rounded-full bg-background/80 hover:bg-background transition-colors opacity-0 group-hover:opacity-100"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="mt-3 p-3 rounded-lg bg-warning/10 border border-warning/20 flex items-start gap-3">
            <FileWarning className="w-4 h-4 text-warning flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-xs font-medium text-warning">Pending Verification</p>
              <p className="text-xs text-muted-foreground">Run fraud detection to verify claim</p>
            </div>
          </div>
        </div>
      )}

      <Button 
        onClick={onAnalyze}
        disabled={!uploadedFile || isAnalyzing}
        className="w-full mt-4 h-11 font-semibold bg-gradient-to-r from-primary to-forensic-cyan hover:opacity-90 transition-opacity"
      >
        {isAnalyzing ? (
          <span className="flex items-center gap-2">
            <div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
            Verifying Claim...
          </span>
        ) : (
          "Run TRUEFY Verification"
        )}
      </Button>
    </div>
  );
};

export default ComplaintUpload;
