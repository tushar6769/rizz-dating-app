/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import express from 'express';
import path from 'path';
import { GoogleGenAI, Type } from '@google/genai';
import dotenv from 'dotenv';
import { createServer as createViteServer } from 'vite';

dotenv.config();

const app = express();
const PORT = 3000;

// Lazy-initialize GoogleGenAI SDK to prevent app crashing on startup when key is missing
let aiClient: GoogleGenAI | null = null;

function getAIClient(): GoogleGenAI {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error('GEMINI_API_KEY is missing. Please add it via the Settings panel to enable the AI coach.');
    }
    aiClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  }
  return aiClient;
}

// Parse payloads safely
app.use(express.json({ limit: '12mb' }));

/**
 * 1. SMART REPLY GENERATOR ENDPOINT
 */
app.post('/api/rizzer/generate-reply', async (req, res) => {
  try {
    const { message, context, platform, tone, goal } = req.body;

    if (!message) {
      return res.status(400).json({ error: 'Message input is required.' });
    }

    const systemInstruction = `
      You are an elite, highly calibrated conversational wingman and texting coach specializing in witty, confident, and magnetic texts.
      We need to generate a perfect reaction response to the following incoming text message: "${message}".
      Additional context: "${context || 'None'}"
      Messaging app context/platform: "${platform || 'tinder'}" (Keep in mind the platform vibe)
      Requested tone: "${tone || 'playful'}"
      Goal/Next objective: "${goal || 'keep_going'}"

      Respond strictly with a valid JSON object of matching schema. Do not output any markdown wrapped or triple-ticks.
    `;

    const prompt = `
      Create replies targeting:
      - Tonality matching "${tone}"
      - Meeting objective: "${goal}" (if ask_out is requested, propose a super low pressure, charming date plan)
      - Never sound desperate, formal, or long-winded.
      - Keep sentences short, authentic, lowercase-welcoming, or teasing.

      Structure the response as:
      {
        "bestReply": "The absolute prime witty/vibrant output string response",
        "whyItWorks": "A brief explanation of why this reply is effective",
        "riskLevel": "Safe" | "Medium" | "Bold",
        "attractionScore": number, // an integer from 1 to 10
        "alternatives": [
          { "text": "Alternative option A", "type": "A descriptive label of this variant's personality tone" },
          { "text": "Alternative option B", "type": "Another descriptive label" },
          { "text": "Alternative option C", "type": "Another descriptive label" }
        ]
      }
    `;

    const response = await getAIClient().models.generateContent({
      model: 'gemini-3.5-flash',
      contents: prompt,
      config: {
        systemInstruction,
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          required: ['bestReply', 'whyItWorks', 'riskLevel', 'attractionScore', 'alternatives'],
          properties: {
            bestReply: { type: Type.STRING, description: 'The absolute best generated reply text.' },
            whyItWorks: { type: Type.STRING, description: 'Brief social-psychology reason why it works.' },
            riskLevel: { type: Type.STRING, description: 'Risk level classification: Safe, Medium, or Bold' },
            attractionScore: { type: Type.INTEGER, description: 'Score out of 10 indicating magnetic tension.' },
            alternatives: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                required: ['text', 'type'],
                properties: {
                  text: { type: Type.STRING },
                  type: { type: Type.STRING },
                },
              },
            },
          },
        },
      },
    });

    const parsedData = JSON.parse(response.text || '{}');
    return res.json(parsedData);
  } catch (error: any) {
    console.error('Error generating replies:', error);
    return res.status(500).json({ error: error.message || 'Error executing AI reply calibrations.' });
  }
});

/**
 * 2. CHAT VIBE & FLIRTING ANALYZER ENDPOINT (Handles OCR / Screenshots & Text History)
 */
app.post('/api/rizzer/analyze-chat', async (req, res) => {
  try {
    const { convoHistory, imageBase64, imageMimeType } = req.body;

    let contents: any[] = [];
    let systemInstruction = `
      You are RizzAI's text-vision systems coach. Your role is to analyze conversational dynamics from a screenshot transcript or typed log.
      Evaluate timing signs, size variations, double texts, conversational interest, indicators of indifference, or dry responses.
      Generate standard metrics, spotting both green flags (invested, playful teasing, typing fast) and red flags (one word, taking ages, ignoring prompts).

      Respond strictly with a valid JSON object matching the required schema. No conversational preamble. No Markdown code block packaging.
    `;

    if (imageBase64 && imageMimeType) {
      contents.push({
        inlineData: {
          mimeType: imageMimeType,
          data: imageBase64,
        },
      });
      contents.push({
        text: 'Analyze the conversation screenshot attached. Read all text bubbles, identify who is speaking, and calculate metrics. Deliver next tactical move prompt.',
      });
    } else if (convoHistory) {
      contents.push({
        text: `Analyze this conversational exchange log:\n"${convoHistory}"`,
      });
    } else {
      return res.status(400).json({ error: 'Please submit either a text history logs or screenshot image.' });
    }

    const response = await getAIClient().models.generateContent({
      model: 'gemini-3.5-flash',
      contents: contents,
      config: {
        systemInstruction,
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          required: [
            'interestScore',
            'attractionLevel',
            'boredomDetected',
            'mixedSignals',
            'flirtingIndicators',
            'conversationMistakes',
            'greenFlags',
            'redFlags',
            'suggestedDirection',
            'nextBestMove',
          ],
          properties: {
            interestScore: { type: Type.INTEGER, description: 'Calculated emotional investment percentage (0-100).' },
            attractionLevel: { type: Type.STRING, description: 'High, Medium, or Low interest evaluation.' },
            boredomDetected: { type: Type.BOOLEAN, description: 'Flag true if dry/low responses are persistent.' },
            mixedSignals: { type: Type.BOOLEAN, description: 'Flag true if signals alternate hot & cold.' },
            flirtingIndicators: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: 'Micro-tension, playful language, emojis, etc.',
            },
            conversationMistakes: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: 'Critique areas (e.g., asking formal interviews, overtexting).',
            },
            greenFlags: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: 'Clear signals of conversational commitment.',
            },
            redFlags: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: 'Indicators of indifference, dryness, or hesitation.',
            },
            suggestedDirection: {
              type: Type.STRING,
              description: 'Short coaching tactical context overview.',
            },
            nextBestMove: {
              type: Type.STRING,
              description: 'A literal copy-paste ready text string they can write to reset/escalate.',
            },
          },
        },
      },
    });

    const parsedData = JSON.parse(response.text || '{}');
    return res.json(parsedData);
  } catch (error: any) {
    console.error('Error analyzing chat vibe:', error);
    return res.status(500).json({ error: error.message || 'Error processing screenshot analyze.' });
  }
});

/**
 * 3. EMERGENCY RESUSCITAION ENDPOINT
 */
app.post('/api/rizzer/revive-chat', async (req, res) => {
  try {
    const { situationType, lastExchangeNotes } = req.body;

    const prompt = `
      We need to resuscitate a stalled text conversation.
      Stall classification: "${situationType || 'Left on Read'}"
      Behind-the-scenes notes: "${lastExchangeNotes || 'Stalled momentum'}"

      Draft high-comedy, playful resets that shift the conversational frame away from defensive/needy to zero-stress. Avoid bitter or manipulative energy. Respond strictly in JSON format.
    `;

    const response = await getAIClient().models.generateContent({
      model: 'gemini-3.5-flash',
      contents: prompt,
      config: {
        systemInstruction: `You are the ultimate emergency medic for texting. Your style is playful, absurd, highly humorous, self-aware, and low key. Output high value, non-needy opener resuscitations.`,
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          required: ['situationSummary', 'bestReply', 'alternatives', 'strategicAdvice'],
          properties: {
            situationSummary: { type: Type.STRING, description: 'Summary analysis of why the chat is stalled.' },
            bestReply: { type: Type.STRING, description: 'The recommended text opener to send.' },
            alternatives: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                required: ['text', 'type', 'riskLevel'],
                properties: {
                  text: { type: Type.STRING },
                  type: { type: Type.STRING, description: 'absurd, self-aware, casual, curiosity, etc.' },
                  riskLevel: { type: Type.STRING, description: 'Safe, Medium, Bold' },
                },
              },
            },
            strategicAdvice: { type: Type.STRING, description: 'Explain why this particular frame resets the vibe.' },
          },
        },
      },
    });

    const parsedData = JSON.parse(response.text || '{}');
    return res.json(parsedData);
  } catch (error: any) {
    console.error('Error reviving chat:', error);
    return res.status(500).json({ error: error.message || 'Failed resuscitating conversation.' });
  }
});

/**
 * 4. OPENER GENERATOR ENDPOINT
 */
app.post('/api/rizzer/generate-opener', async (req, res) => {
  try {
    const { datingBio, interests, photosDescription } = req.body;

    const systemInstruction = `
      You are RizzAI's expert high-priority first-text engineer.
      Your responsibility is to take background variables of their profile and formulate the most fascinating, clever, funny, and non-generic openers.
      Never say: "Hey! How's your weekend going?", "Cute smile :)", or similar cringe things.
      Focus on playful challenges, absurd observations, or micro-theories about their interests.
    `;

    const prompt = `
      Create 4 standout, creative openers based on:
      Target Bio Context: "${datingBio || 'None Specified'}"
      Target Interests: "${interests || 'None Specified'}"
      Photos Description: "${photosDescription || 'None Specified'}"

      Supply ideas categorized in distinct personality vectors. Respond in schema JSON format.
    `;

    const response = await getAIClient().models.generateContent({
      model: 'gemini-3.5-flash',
      contents: prompt,
      config: {
        systemInstruction,
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          required: ['openers', 'coachingPointer'],
          properties: {
            openers: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                required: ['category', 'text', 'deliveryTip'],
                properties: {
                  category: { type: Type.STRING, description: 'e.g., Clever Opener, Funny Starter, Teasing Challenge, Absurd Observation' },
                  text: { type: Type.STRING, description: 'The exact copy-paste ready icebreaker text.' },
                  deliveryTip: { type: Type.STRING, description: 'Short context tip on how to carry the conversation.' },
                },
              },
            },
            coachingPointer: { type: Type.STRING, description: 'Core strategic advice why these fit the target profile.' },
          },
        },
      },
    });

    const parsed = JSON.parse(response.text || '{}');
    return res.json(parsed);
  } catch (error: any) {
    console.error('Error in generate opener:', error);
    return res.status(500).json({ error: error.message || 'Opener formulation stalled.' });
  }
});

/**
 * 5. DATINGS BIO WRITER ENDPOINT
 */
app.post('/api/rizzer/generate-bio', async (req, res) => {
  try {
    const { characterTraits, hobbies, occupation, selectedStyles } = req.body;

    const prompt = `
      Draft standout dating bios for dating apps (Tinder, Hinge, Bumble) based on:
      - Character Traits/Vibe: "${characterTraits || 'charming, curious'}"
      - Hobbies/Passions: "${hobbies || 'coffee, vinyl, travel'}"
      - Occupation/Daily grind: "${occupation || 'Designer'}"
      - Requested Templates/Styles to highlight: [${selectedStyles?.join(', ') || 'funny, luxury, soft aesthetic'}]

      Each bio should be visually inviting, include short spaces, hooks, confidence, and subtle call to action.
      Provide multiple varied options in schema JSON.
    `;

    const response = await getAIClient().models.generateContent({
      model: 'gemini-3.5-flash',
      contents: prompt,
      config: {
        systemInstruction: `You are an elite dating bio copywriter. You write short, highly aesthetic, distinct profiles with personality punchlines. Zero copy-paste template clichés. Standout charisma only.`,
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          required: ['bios'],
          properties: {
            bios: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                required: ['templateName', 'bioText', 'vibeDescription'],
                properties: {
                  templateName: { type: Type.STRING, description: 'Funny, Luxury lifestyle, Soft aesthetic, Gym personality, Confident alpha, Mysterious, Intelligent/chill' },
                  bioText: { type: Type.STRING, description: 'Dating bio content with perfect formatting & styling.' },
                  vibeDescription: { type: Type.STRING, description: 'Brief look at who this bio attracts.' },
                },
              },
            },
          },
        },
      },
    });

    const parsed = JSON.parse(response.text || '{}');
    return res.json(parsed);
  } catch (error: any) {
    console.error('Error writing bios:', error);
    return res.status(500).json({ error: error.message || 'Bio creator system error.' });
  }
});

// Configure Vite integration for Dev or Production Static serving
async function bootstrap() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`RizzAI server running correctly on port ${PORT}`);
  });
}

bootstrap();
