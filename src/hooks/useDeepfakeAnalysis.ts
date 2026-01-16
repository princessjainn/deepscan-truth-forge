import { useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";

export interface AnalysisResult {
  verdict: "LIKELY_AUTHENTIC" | "LIKELY_MANIPULATED" | "INCONCLUSIVE";
  confidence: number;
  fakeProbability: number;
  manipulationTypes?: string[];
  riskLevel: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  visualAnalysis?: {
    faceSwapScore: number;
    ganArtifactScore: number;
    lightingConsistency: number;
    boundaryArtifacts: number;
    details: string;
  };
  audioAnalysis?: {
    voiceAuthenticity: number;
    voiceCloningScore: number;
    lipSyncAccuracy: number;
    spectralAnomaly: number;
    details: string;
  };
  temporalAnalysis?: {
    frameConsistency: number;
    blinkPatternScore: number;
    motionCoherence: number;
    details: string;
  };
  metadataAnalysis?: {
    exifIntegrity: number;
    sourceVerified: boolean;
    editingDetected: boolean;
    details: string;
  };
  threats?: Array<{
    type: string;
    severity: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
    description: string;
  }>;
  forensicSummary?: string;
  recommendations?: string[];
  error?: string;
}

interface UseDeepfakeAnalysisReturn {
  isAnalyzing: boolean;
  analysisResult: AnalysisResult | null;
  analyzeMedia: (
    mediaType: "image" | "video" | "audio",
    mediaData?: string,
    analysisModules?: string[]
  ) => Promise<void>;
  resetAnalysis: () => void;
}

export const useDeepfakeAnalysis = (): UseDeepfakeAnalysisReturn => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const { toast } = useToast();

  const analyzeMedia = async (
    mediaType: "image" | "video" | "audio",
    mediaData?: string,
    analysisModules: string[] = ["visual", "audio", "temporal", "metadata"]
  ) => {
    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      const { data, error } = await supabase.functions.invoke("analyze-media", {
        body: {
          mediaType,
          mediaData,
          analysisModules,
        },
      });

      if (error) {
        throw error;
      }

      if (data.error) {
        if (data.error.includes("Rate limit")) {
          toast({
            title: "Rate Limit Exceeded",
            description: "Please wait a moment before trying again.",
            variant: "destructive",
          });
        } else if (data.error.includes("Payment required")) {
          toast({
            title: "Credits Required",
            description: "Please add credits to your Lovable AI workspace.",
            variant: "destructive",
          });
        } else {
          throw new Error(data.error);
        }
        return;
      }

      setAnalysisResult(data);
      
      toast({
        title: "Analysis Complete",
        description: `Verdict: ${data.verdict.replace("_", " ")} (${data.confidence}% confidence)`,
        variant: data.verdict === "LIKELY_AUTHENTIC" ? "default" : "destructive",
      });

    } catch (error) {
      console.error("Analysis failed:", error);
      toast({
        title: "Analysis Failed",
        description: error instanceof Error ? error.message : "An error occurred during analysis",
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetAnalysis = () => {
    setAnalysisResult(null);
    setIsAnalyzing(false);
  };

  return {
    isAnalyzing,
    analysisResult,
    analyzeMedia,
    resetAnalysis,
  };
};
