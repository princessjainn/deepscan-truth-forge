import { useState } from "react";
import Header from "@/components/Header";
import MediaUpload from "@/components/MediaUpload";
import RiskSummary from "@/components/RiskSummary";
import VisualForensics from "@/components/VisualForensics";
import AudioAnalysis from "@/components/AudioAnalysis";
import TemporalAnalysis from "@/components/TemporalAnalysis";
import MetadataAnalysis from "@/components/MetadataAnalysis";
import ForensicReport from "@/components/ForensicReport";
import IntegrationFlow from "@/components/IntegrationFlow";
import Footer from "@/components/Footer";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Eye, Mic, Clock, FileCode } from "lucide-react";
import { useDeepfakeAnalysis, AnalysisResult } from "@/hooks/useDeepfakeAnalysis";

const Index = () => {
  const [mediaType, setMediaType] = useState<"image" | "video" | "audio">("video");
  const { isAnalyzing, analysisResult, analyzeMedia, resetAnalysis } = useDeepfakeAnalysis();

  const handleAnalyze = async (type: "image" | "video" | "audio", mediaData?: string) => {
    setMediaType(type);
    await analyzeMedia(type, mediaData, ["visual", "audio", "temporal", "metadata"]);
  };

  const handleReset = () => {
    resetAnalysis();
  };

  const isAnalyzed = !!analysisResult;

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Header />
      
      <main className="flex-1 container mx-auto px-4 lg:px-8 py-8 space-y-8">
        
        {/* Section 1: Analysis Flow */}
        <section>
          <IntegrationFlow isAnalyzed={isAnalyzed} isAnalyzing={isAnalyzing} />
        </section>

        {/* Section 2: Main Analysis Panel */}
        <section>
          <div className="flex items-center gap-2 mb-4">
            <div className="w-1 h-6 bg-primary rounded-full" />
            <h2 className="text-lg font-semibold">AI-Powered Media Authenticity Analysis</h2>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left - Upload Panel */}
            <div className="lg:col-span-4">
              <MediaUpload 
                onAnalyze={handleAnalyze} 
                isAnalyzing={isAnalyzing}
                onReset={handleReset}
              />
            </div>

            {/* Right - Risk Summary */}
            <div className="lg:col-span-8">
              <RiskSummary 
                isAnalyzed={isAnalyzed} 
                isAnalyzing={isAnalyzing}
                analysisResult={analysisResult}
              />
            </div>
          </div>
        </section>

        {/* Section 3: Detailed Analysis Tabs */}
        <section>
          <div className="flex items-center gap-2 mb-4">
            <div className="w-1 h-6 bg-primary rounded-full" />
            <h2 className="text-lg font-semibold">Multi-Modal Forensic Analysis</h2>
          </div>

          <Tabs defaultValue="visual" className="w-full">
            <TabsList className="w-full justify-start bg-card border border-border p-1 h-auto flex-wrap gap-1">
              <TabsTrigger 
                value="visual" 
                className="flex items-center gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
              >
                <Eye className="w-4 h-4" />
                <span>Visual</span>
              </TabsTrigger>
              <TabsTrigger 
                value="audio" 
                className="flex items-center gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
              >
                <Mic className="w-4 h-4" />
                <span>Audio</span>
              </TabsTrigger>
              <TabsTrigger 
                value="temporal" 
                className="flex items-center gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
              >
                <Clock className="w-4 h-4" />
                <span>Temporal</span>
              </TabsTrigger>
              <TabsTrigger 
                value="metadata" 
                className="flex items-center gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
              >
                <FileCode className="w-4 h-4" />
                <span>Metadata</span>
              </TabsTrigger>
            </TabsList>

            <div className="mt-4">
              <TabsContent value="visual" className="mt-0">
                <VisualForensics isAnalyzed={isAnalyzed} analysisResult={analysisResult} />
              </TabsContent>
              
              <TabsContent value="audio" className="mt-0">
                <AudioAnalysis isAnalyzed={isAnalyzed} analysisResult={analysisResult} />
              </TabsContent>
              
              <TabsContent value="temporal" className="mt-0">
                <TemporalAnalysis isAnalyzed={isAnalyzed} analysisResult={analysisResult} />
              </TabsContent>

              <TabsContent value="metadata" className="mt-0">
                <MetadataAnalysis isAnalyzed={isAnalyzed} analysisResult={analysisResult} />
              </TabsContent>
            </div>
          </Tabs>
        </section>

        {/* Section 4: Forensic Report */}
        <section>
          <div className="flex items-center gap-2 mb-4">
            <div className="w-1 h-6 bg-primary rounded-full" />
            <h2 className="text-lg font-semibold">Explainable Forensic Report</h2>
          </div>
          
          <ForensicReport isAnalyzed={isAnalyzed} analysisResult={analysisResult} />
        </section>

      </main>

      <Footer />
    </div>
  );
};

export default Index;
