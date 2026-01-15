import { useState } from "react";
import Header from "@/components/Header";
import ComplaintUpload from "@/components/ComplaintUpload";
import FraudRiskScore from "@/components/FraudRiskScore";
import MediaReuseDetection from "@/components/MediaReuseDetection";
import OrderConsistencyCheck from "@/components/OrderConsistencyCheck";
import VoiceFraudDetection from "@/components/VoiceFraudDetection";
import BusinessImpactDashboard from "@/components/BusinessImpactDashboard";
import IntegrationFlow from "@/components/IntegrationFlow";
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
        {/* Integration Flow */}
        <div className="mb-6">
          <IntegrationFlow isAnalyzed={isAnalyzed} isAnalyzing={isAnalyzing} />
        </div>

        {/* Main grid layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* Left column - Complaint Upload */}
          <div className="lg:col-span-3">
            <ComplaintUpload onAnalyze={handleAnalyze} isAnalyzing={isAnalyzing} />
          </div>

          {/* Center column - Fraud Risk Score */}
          <div className="lg:col-span-6">
            <FraudRiskScore isAnalyzed={isAnalyzed} isAnalyzing={isAnalyzing} />
          </div>

          {/* Right column - Order Consistency */}
          <div className="lg:col-span-3">
            <OrderConsistencyCheck isAnalyzed={isAnalyzed} />
          </div>
        </div>

        {/* Secondary analysis grid */}
        <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <MediaReuseDetection isAnalyzed={isAnalyzed} />
          <VoiceFraudDetection isAnalyzed={isAnalyzed} />
        </div>

        {/* Business Impact Dashboard - Full width */}
        <div className="mt-6">
          <BusinessImpactDashboard isAnalyzed={isAnalyzed} />
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Index;
