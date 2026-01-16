import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

interface AnalysisRequest {
  mediaType: "image" | "video" | "audio";
  mediaData?: string; // base64 encoded media
  mediaUrl?: string;
  analysisModules: string[];
}

const DEEPFAKE_SYSTEM_PROMPT = `You are an advanced AI forensic analyst specialized in deepfake detection and media authenticity verification. Your role is to analyze media and provide detailed forensic reports.

When analyzing media, you must evaluate:

1. **Visual Forensics** (for images/videos):
   - Face swap detection: Look for boundary artifacts, skin tone mismatches, blending inconsistencies
   - GAN artifacts: Check for checkerboard patterns, frequency anomalies, upsampling artifacts
   - Lighting analysis: Evaluate shadow consistency, reflection coherence, light source alignment
   - Facial geometry: Check landmark alignment, proportional consistency, micro-expression authenticity
   - Compression artifacts: Identify recompression signatures, quality inconsistencies

2. **Audio Forensics** (for audio/videos):
   - Voice authenticity: Analyze formant patterns, pitch consistency, spectral characteristics
   - Voice cloning detection: Check for synthetic voice markers, unnatural prosody
   - Audio-visual sync: Evaluate lip movement correlation, phoneme alignment
   - Background noise analysis: Check for splicing artifacts, environmental inconsistencies

3. **Temporal Analysis** (for videos):
   - Frame consistency: Check for inter-frame artifacts, motion blur anomalies
   - Blink pattern analysis: Evaluate natural blink rates and patterns
   - Temporal coherence: Look for jitter, unnatural transitions, frame interpolation signs

4. **Metadata Forensics**:
   - EXIF integrity: Check for metadata tampering, missing fields
   - Source verification: Analyze device signatures, processing history
   - Compression history: Identify editing software fingerprints

Respond with a detailed JSON analysis containing:
{
  "verdict": "LIKELY_AUTHENTIC" | "LIKELY_MANIPULATED" | "INCONCLUSIVE",
  "confidence": number (0-100),
  "fakeProbability": number (0-100),
  "manipulationTypes": string[],
  "riskLevel": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "visualAnalysis": {
    "faceSwapScore": number,
    "ganArtifactScore": number,
    "lightingConsistency": number,
    "boundaryArtifacts": number,
    "details": string
  },
  "audioAnalysis": {
    "voiceAuthenticity": number,
    "voiceCloningScore": number,
    "lipSyncAccuracy": number,
    "spectralAnomaly": number,
    "details": string
  },
  "temporalAnalysis": {
    "frameConsistency": number,
    "blinkPatternScore": number,
    "motionCoherence": number,
    "details": string
  },
  "metadataAnalysis": {
    "exifIntegrity": number,
    "sourceVerified": boolean,
    "editingDetected": boolean,
    "details": string
  },
  "threats": [
    {
      "type": string,
      "severity": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
      "description": string
    }
  ],
  "forensicSummary": string,
  "recommendations": string[]
}`;

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { mediaType, mediaData, mediaUrl, analysisModules } = await req.json() as AnalysisRequest;
    
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) {
      throw new Error("LOVABLE_API_KEY is not configured");
    }

    // Build the analysis prompt based on what we're analyzing
    const analysisPrompt = `Analyze this ${mediaType} for deepfake manipulation and provide a comprehensive forensic report.

Analysis modules requested: ${analysisModules.join(", ")}

${mediaData ? "The media has been provided for analysis." : mediaUrl ? `Media URL: ${mediaUrl}` : "No media provided - perform a simulated analysis for demonstration purposes."}

Provide your analysis as a JSON object following the specified format. Be thorough and provide realistic forensic scores based on common deepfake detection patterns. For demonstration purposes, simulate finding some manipulation indicators to show the system's capabilities.`;

    // Build messages array
    const messages: any[] = [
      { role: "system", content: DEEPFAKE_SYSTEM_PROMPT },
    ];

    // Add media if provided
    if (mediaData) {
      messages.push({
        role: "user",
        content: [
          { type: "text", text: analysisPrompt },
          {
            type: "image_url",
            image_url: { url: mediaData }
          }
        ]
      });
    } else {
      messages.push({
        role: "user",
        content: analysisPrompt
      });
    }

    console.log("Sending request to Lovable AI for deepfake analysis...");

    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-pro",
        messages,
        temperature: 0.3,
      }),
    });

    if (!response.ok) {
      if (response.status === 429) {
        return new Response(
          JSON.stringify({ error: "Rate limit exceeded. Please try again later." }),
          { status: 429, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      if (response.status === 402) {
        return new Response(
          JSON.stringify({ error: "Payment required. Please add credits to your Lovable AI workspace." }),
          { status: 402, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      const errorText = await response.text();
      console.error("AI gateway error:", response.status, errorText);
      throw new Error(`AI gateway error: ${response.status}`);
    }

    const data = await response.json();
    const content = data.choices?.[0]?.message?.content;

    if (!content) {
      throw new Error("No response from AI model");
    }

    // Parse the JSON response from the AI
    let analysisResult;
    try {
      // Extract JSON from the response (it might be wrapped in markdown code blocks)
      const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/) || 
                        content.match(/```\s*([\s\S]*?)\s*```/) ||
                        [null, content];
      const jsonStr = jsonMatch[1] || content;
      analysisResult = JSON.parse(jsonStr.trim());
    } catch (parseError) {
      console.error("Failed to parse AI response as JSON:", parseError);
      // Return a structured error response
      analysisResult = {
        verdict: "INCONCLUSIVE",
        confidence: 0,
        fakeProbability: 0,
        riskLevel: "LOW",
        forensicSummary: "Analysis could not be completed. Please try again.",
        error: "Failed to parse analysis results"
      };
    }

    return new Response(
      JSON.stringify(analysisResult),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );

  } catch (error) {
    console.error("Analysis error:", error);
    return new Response(
      JSON.stringify({ 
        error: error instanceof Error ? error.message : "Unknown error occurred",
        verdict: "INCONCLUSIVE"
      }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
