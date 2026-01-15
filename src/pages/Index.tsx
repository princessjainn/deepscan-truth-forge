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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Fingerprint, Package, Mic, BarChart3 } from "lucide-react";

const Index = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isAnalyzed, setIsAnalyzed] = useState(false);

  const handleAnalyze = () => {
    setIsAnalyzing(true);
    setIsAnalyzed(false);
    
    setTimeout(() => {
      setIsAnalyzing(false);
      setIsAnalyzed(true);
    }, 3000);
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Header />
      
      <main className="flex-1 container mx-auto px-4 lg:px-8 py-8 space-y-8">
        
        {/* Section 1: Integration Flow */}
        <section>
          <IntegrationFlow isAnalyzed={isAnalyzed} isAnalyzing={isAnalyzing} />
        </section>

        {/* Section 2: Main Verification Panel */}
        <section>
          <div className="flex items-center gap-2 mb-4">
            <div className="w-1 h-6 bg-primary rounded-full" />
            <h2 className="text-lg font-semibold">Complaint Verification</h2>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left - Upload Panel */}
            <div className="lg:col-span-4">
              <ComplaintUpload onAnalyze={handleAnalyze} isAnalyzing={isAnalyzing} />
            </div>

            {/* Right - Fraud Risk Score */}
            <div className="lg:col-span-8">
              <FraudRiskScore isAnalyzed={isAnalyzed} isAnalyzing={isAnalyzing} />
            </div>
          </div>
        </section>

        {/* Section 3: Detailed Analysis Tabs */}
        <section>
          <div className="flex items-center gap-2 mb-4">
            <div className="w-1 h-6 bg-primary rounded-full" />
            <h2 className="text-lg font-semibold">Detailed Analysis</h2>
          </div>

          <Tabs defaultValue="consistency" className="w-full">
            <TabsList className="w-full justify-start bg-card border border-border p-1 h-auto flex-wrap gap-1">
              <TabsTrigger 
                value="consistency" 
                className="flex items-center gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
              >
                <Package className="w-4 h-4" />
                <span>Order Consistency</span>
              </TabsTrigger>
              <TabsTrigger 
                value="reuse" 
                className="flex items-center gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
              >
                <Fingerprint className="w-4 h-4" />
                <span>Media Reuse</span>
              </TabsTrigger>
              <TabsTrigger 
                value="voice" 
                className="flex items-center gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
              >
                <Mic className="w-4 h-4" />
                <span>Voice Analysis</span>
              </TabsTrigger>
              <TabsTrigger 
                value="impact" 
                className="flex items-center gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
              >
                <BarChart3 className="w-4 h-4" />
                <span>Business Impact</span>
              </TabsTrigger>
            </TabsList>

            <div className="mt-4">
              <TabsContent value="consistency" className="mt-0">
                <OrderConsistencyCheck isAnalyzed={isAnalyzed} />
              </TabsContent>
              
              <TabsContent value="reuse" className="mt-0">
                <MediaReuseDetection isAnalyzed={isAnalyzed} />
              </TabsContent>
              
              <TabsContent value="voice" className="mt-0">
                <VoiceFraudDetection isAnalyzed={isAnalyzed} />
              </TabsContent>

              <TabsContent value="impact" className="mt-0">
                <BusinessImpactDashboard isAnalyzed={isAnalyzed} />
              </TabsContent>
            </div>
          </Tabs>
        </section>

      </main>

      <Footer />
    </div>
  );
};

export default Index;
