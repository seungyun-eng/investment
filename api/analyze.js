import { generateObject } from 'ai';
import { z } from 'zod';

const Result = z.object({
  valid: z.boolean(),
  score: z.number().min(0).max(100),
  english: z.number().min(0).max(100),
  structure: z.number().min(0).max(100),
  judgment: z.number().min(0).max(100),
  executive_presence: z.number().min(0).max(100),
  verdict: z.string(),
  strengths: z.array(z.string()),
  weaknesses: z.array(z.string()),
  natural_version: z.string(),
  executive_version: z.string(),
  challenge_question: z.string(),
  explanation_ko: z.string()
});

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });
  const { mode = 'mba', caseData = null, prompt = '', answer = '' } = req.body || {};
  if (typeof answer !== 'string' || !answer.trim()) return res.status(400).json({ error: 'Missing answer' });

  const system = `You are WorkSpeak, a rigorous English and business communication coach for a Korean technical professional preparing for MBA-level and executive communication.
Evaluate semantic content, not keyword presence. Never praise meaningless, irrelevant, unsupported, or extremely short answers.
If an answer is nonsensical, unrelated, a fragment, or too short to evaluate, set valid=false, give very low scores (0-20), use no strengths, and clearly explain what is missing.
For MBA mode, compare the user's answer against the actual case facts, available choices, risks, and trade-offs. A fluent but poorly reasoned answer must receive a low judgment score.
For speaking mode, evaluate meaning, grammar, naturalness, precision, professional tone, and whether the main point is clear.
For drill mode, compare the intended prompt meaning with the user's answer and flag semantic as well as grammatical errors.
Return verdict, weaknesses, and explanation in Korean. Keep natural_version and executive_version in English. Be specific rather than generic.`;

  try {
    const { object } = await generateObject({
      model: 'openai/gpt-5.6-sol',
      schema: Result,
      system,
      prompt: JSON.stringify({ mode, caseData, prompt, answer })
    });
    return res.status(200).json(object);
  } catch (error) {
    console.error('WORKSPEAK_AI_ERROR', error);
    return res.status(500).json({ error: 'AI analysis failed' });
  }
}
