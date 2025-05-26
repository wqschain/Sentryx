export interface SentimentResult {
  sentiment: string;
  confidence: number;
  is_relevant: boolean;
  relevance_score: number;
  relevance_explanation: string;
  matched_terms: Record<string, string[]>;
  feedback?: string;
}

export interface SentimentError {
  message: string;
  feedback?: string;
} 