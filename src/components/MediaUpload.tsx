import { useState, useCallback } from "react";
import { Upload, Image, Video, Music, FileWarning, Play, X } from "lucide-react";
import { Button } from "@/components/ui/button";

interface MediaUploadProps {
  onAnalyze: () => void;
  isAnalyzing: boolean;
}

const MediaUpload = ({ onAnalyze, isAnalyzing }: MediaUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<{ name: string; type: string; preview?: string } | null>(null);

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
    
    // Simulate file upload
    setUploadedFile({
      name: "suspect_media_001.mp4",
      type: "video",
      preview: undefined
    });
  }, []);

  const handleFileSelect = () => {
    // Simulate file selection
    setUploadedFile({
      name: "suspect_media_001.mp4",
      type: "video",
      preview: undefined
    });
  };

  const clearUpload = () => {
    setUploadedFile(null);
  };

  return (
    <div className="forensic-card p-6 h-full flex flex-col">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-lg bg-primary/10">
          <Upload className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h2 className="font-semibold">Media Input</h2>
          <p className="text-xs text-muted-foreground">Upload media for forensic analysis</p>
        </div>
      </div>

      {!uploadedFile ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={handleFileSelect}
          className={`
            flex-1 border-2 border-dashed rounded-lg p-8 
            flex flex-col items-center justify-center gap-4 cursor-pointer
            transition-all duration-300
            ${isDragging 
              ? "border-primary bg-primary/5" 
              : "border-border hover:border-primary/50 hover:bg-muted/30"
            }
          `}
        >
          <div className={`
            p-4 rounded-full bg-muted transition-all duration-300
            ${isDragging ? "scale-110 bg-primary/20" : ""}
          `}>
            <Upload className={`w-8 h-8 ${isDragging ? "text-primary" : "text-muted-foreground"}`} />
          </div>
          
          <div className="text-center">
            <p className="font-medium mb-1">
              {isDragging ? "Drop file here" : "Drag & drop media file"}
            </p>
            <p className="text-sm text-muted-foreground">
              or click to browse
            </p>
          </div>

          <div className="flex flex-wrap justify-center gap-3 mt-2">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground bg-muted px-3 py-1.5 rounded-full">
              <Image className="w-3.5 h-3.5" />
              <span>Images</span>
            </div>
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground bg-muted px-3 py-1.5 rounded-full">
              <Video className="w-3.5 h-3.5" />
              <span>Videos</span>
            </div>
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground bg-muted px-3 py-1.5 rounded-full">
              <Music className="w-3.5 h-3.5" />
              <span>Audio</span>
            </div>
          </div>

          <p className="text-xs text-muted-foreground mt-2">
            Supported: MP4, AVI, MOV, JPG, PNG, WEBP, MP3, WAV • Max 500MB
          </p>
        </div>
      ) : (
        <div className="flex-1 flex flex-col">
          <div className="flex-1 bg-muted/50 rounded-lg relative overflow-hidden group">
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="w-20 h-20 mx-auto mb-4 rounded-xl bg-forensic-navy flex items-center justify-center">
                  <Video className="w-10 h-10 text-primary" />
                </div>
                <p className="font-mono text-sm">{uploadedFile.name}</p>
                <p className="text-xs text-muted-foreground mt-1">Video file • 24.5 MB</p>
              </div>
            </div>
            
            <button 
              onClick={clearUpload}
              className="absolute top-3 right-3 p-2 rounded-full bg-background/80 hover:bg-background transition-colors opacity-0 group-hover:opacity-100"
            >
              <X className="w-4 h-4" />
            </button>

            <button className="absolute bottom-3 left-1/2 -translate-x-1/2 flex items-center gap-2 px-4 py-2 rounded-full bg-background/80 hover:bg-background transition-colors opacity-0 group-hover:opacity-100">
              <Play className="w-4 h-4" />
              <span className="text-sm">Preview</span>
            </button>
          </div>

          <div className="mt-4 p-3 rounded-lg bg-warning/10 border border-warning/20 flex items-start gap-3">
            <FileWarning className="w-5 h-5 text-warning flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-warning">Unverified Media</p>
              <p className="text-xs text-muted-foreground">Run forensic analysis to verify authenticity</p>
            </div>
          </div>
        </div>
      )}

      <Button 
        onClick={onAnalyze}
        disabled={!uploadedFile || isAnalyzing}
        className="w-full mt-6 h-12 text-base font-semibold bg-gradient-to-r from-primary to-forensic-cyan hover:opacity-90 transition-opacity"
      >
        {isAnalyzing ? (
          <span className="flex items-center gap-2">
            <div className="w-5 h-5 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
            Running Forensic Analysis...
          </span>
        ) : (
          "Run Forensic Analysis"
        )}
      </Button>
    </div>
  );
};

export default MediaUpload;
