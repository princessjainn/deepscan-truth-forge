import { useState } from "react";
import MainLayout from "@/components/layouts/MainLayout";
import ForensicReport from "@/components/ForensicReport";
import MediaUpload from "@/components/MediaUpload";
import { useDeepfakeAnalysis } from "@/hooks/useDeepfakeAnalysis";
import { FileText, Upload, Download, Share2 } from "lucide-react";
import { Button } from "@/components/ui/button";

const ReportsPage = () => {
  const [mediaType, setMediaType] = useState<"image" | "video" | "audio">("video");
  const { isAnalyzing, analysisResult, analyzeMedia, resetAnalysis } = useDeepfakeAnalysis();

  const handleAnalyze = async (type: "image" | "video" | "audio", mediaData?: string) => {
    setMediaType(type);
    await analyzeMedia(type, mediaData, ["visual", "audio", "temporal", "metadata"]);
  };

  const isAnalyzed = !!analysisResult;

  return (
    <MainLayout title="Reports" subtitle="Forensic Report Generation">
      <div className="space-y-6">
        {/* Upload Section when no analysis */}
        {!isAnalyzed && !isAnalyzing && (
          <div className="max-w-2xl mx-auto">
            <MediaUpload
              onAnalyze={handleAnalyze}
              isAnalyzing={isAnalyzing}
              onReset={resetAnalysis}
            />
          </div>
        )}

        {/* Report Section */}
        {isAnalyzed && (
          <>
            {/* Report Actions */}
            <div className="forensic-card p-4 flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-primary/10">
                  <FileText className="w-5 h-5 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold">Forensic Report Ready</h3>
                  <p className="text-sm text-muted-foreground">
                    Export or share your analysis report
                  </p>
                </div>
              </div>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" className="gap-2">
                  <Share2 className="w-4 h-4" />
                  Share
                </Button>
                <Button size="sm" className="gap-2">
                  <Download className="w-4 h-4" />
                  Export PDF
                </Button>
              </div>
            </div>

            {/* Forensic Report */}
            <ForensicReport isAnalyzed={isAnalyzed} analysisResult={analysisResult} />

            {/* New Analysis Button */}
            <div className="text-center">
              <Button variant="outline" onClick={resetAnalysis} className="gap-2">
                <Upload className="w-4 h-4" />
                Analyze New Media
              </Button>
            </div>
          </>
        )}

        {/* Empty State */}
        {!isAnalyzed && !isAnalyzing && (
          <div className="text-center py-12">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-muted mb-4">
              <FileText className="w-8 h-8 text-muted-foreground" />
            </div>
            <h3 className="text-lg font-semibold mb-2">No Reports Available</h3>
            <p className="text-muted-foreground max-w-md mx-auto">
              Upload and analyze media to generate a comprehensive forensic report
            </p>
          </div>
        )}

        {/* Loading State */}
        {isAnalyzing && (
          <div className="text-center py-12">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mb-4">
              <div className="w-8 h-8 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Generating Report...</h3>
            <p className="text-muted-foreground max-w-md mx-auto">
              Please wait while we analyze your media and generate the forensic report
            </p>
          </div>
        )}
      </div>
    </MainLayout>
  );
};

export default ReportsPage;
