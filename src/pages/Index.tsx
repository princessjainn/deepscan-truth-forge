import { useState } from "react";
import Header from "@/components/Header";
import MediaUpload from "@/components/MediaUpload";
import RiskSummary from "@/components/RiskSummary";
import ManipulationTimeline from "@/components/ManipulationTimeline";
import AnalysisTabs from "@/components/AnalysisTabs";
import HumanPlausibilityIndex from "@/components/HumanPlausibilityIndex";
import ForensicReport from "@/components/ForensicReport";
import RobustnessTest from "@/components/RobustnessTest";
import Footer from "@/components/Footer";

const Index = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isAnalyzed, setIsAnalyzed] = useState(false);

  const handleAnalyze = () => {
    setIsAnalyzing(true);
    setIsAnalyzed(false);
    
    // Simulate analysis time
    setTimeout(() => {
      setIsAnalyzing(false);
      setIsAnalyzed(true);
    }, 3000);
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Header />
      
      <main className="flex-1 container mx-auto px-4 lg:px-6 py-6">
        {/* Main grid layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* Left column - Media Upload */}
          <div className="lg:col-span-3">
            <MediaUpload onAnalyze={handleAnalyze} isAnalyzing={isAnalyzing} />
          </div>

          {/* Center column - Risk Summary & Timeline */}
          <div className="lg:col-span-6 space-y-6">
            <RiskSummary isAnalyzed={isAnalyzed} isAnalyzing={isAnalyzing} />
            <ManipulationTimeline isAnalyzed={isAnalyzed} />
          </div>

          {/* Right column - Human Plausibility Index */}
          <div className="lg:col-span-3">
            <HumanPlausibilityIndex isAnalyzed={isAnalyzed} />
          </div>
        </div>

        {/* Full width sections */}
        <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <AnalysisTabs isAnalyzed={isAnalyzed} />
          <ForensicReport isAnalyzed={isAnalyzed} />
        </div>

        {/* Robustness Test - Full width */}
        <div className="mt-6">
          <RobustnessTest isAnalyzed={isAnalyzed} />
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Index;
