grog cloud
Playground
API Keys
Dashboard
Documentation
Upgrade

Steven 
Personal
Docs
API Reference
Get Started
Overview
Quickstart
OpenAI Compatibility
Models
Rate Limits
Features
Text
Speech to Text
Text to Speech
Reasoning
Vision
Agentic Tooling
Advanced Features
Batch Processing
Flex Processing
Content Moderation
Prefilling
Tool Use
Developer Resources
Groq Libraries
Groq Badge
Examples
Applications Showcase
Resources
Prompting Guide
Integrations
Integrations Catalog
Support & Guidelines
Errors
Changelog
Policies & Notices
Supported Models
GroqCloud currently supports the following models:


Production Models
Note: Production models are intended for use in your production environments. They meet or exceed our high standards for speed, quality, and reliability. Read more here.

MODEL ID	DEVELOPER	CONTEXT WINDOW (TOKENS)	MAX COMPLETION TOKENS	MAX FILE SIZE	DETAILS
gemma2-9b-it
Google
8,192
-
-
Details
llama-3.3-70b-versatile
Meta
128K
32,768
-
Details
llama-3.1-8b-instant
Meta
128K
8,192
-
Details
llama-guard-3-8b
Meta
8,192
-
-
Details
llama3-70b-8192
Meta
8,192
-
-
Details
llama3-8b-8192
Meta
8,192
-
-
Details
whisper-large-v3
OpenAI
-
-
25 MB
Details
whisper-large-v3-turbo
OpenAI
-
-
25 MB
Details
distil-whisper-large-v3-en
HuggingFace
-
-
25 MB
Details

Preview Models
Note: Preview models are intended for evaluation purposes only and should not be used in production environments as they may be discontinued at short notice. Read more about deprecations here.

MODEL ID	DEVELOPER	CONTEXT WINDOW (TOKENS)	MAX COMPLETION TOKENS	MAX FILE SIZE	DETAILS
allam-2-7b
Saudi Data and AI Authority (SDAIA)
4,096
-
-
Details
deepseek-r1-distill-llama-70b
DeepSeek
128K
-
-
Details
meta-llama/llama-4-maverick-17b-128e-instruct
Meta
131,072
8192
-
Details
meta-llama/llama-4-scout-17b-16e-instruct
Meta
131,072
8192
-
Details
mistral-saba-24b
Mistral
32K
-
-
Details
playai-tts
Playht, Inc
10K
-
Details
playai-tts-arabic
Playht, Inc
10K
-
-
Details
qwen-qwq-32b
Alibaba Cloud
128K
-
-
Details

Preview Systems
Systems are a collection of models and tools that work together to answer a user query.

Note: Preview systems are intended for evaluation purposes only and should not be used in production environments as they may be discontinued at short notice. Read more about deprecations here.

MODEL ID	DEVELOPER	CONTEXT WINDOW (TOKENS)	MAX COMPLETION TOKENS	MAX FILE SIZE	DETAILS
compound-beta
Groq
128K
8192
-
Details
compound-beta-mini
Groq
128K
8192
-
Details

Learn More About Agentic Tooling
Discover how to build powerful applications with real-time web search and code execution

Deprecated models are models that are no longer supported or will no longer be supported in the future. See our deprecation guidelines and deprecated models here.


Hosted models are directly accessible through the GroqCloud Models API endpoint using the model IDs mentioned above. You can use the https://api.groq.com/openai/v1/models endpoint to return a JSON list of all active models:

curl
JavaScript
Python

import Groq from "groq-sdk";

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

const getModels = async () => {
  return await groq.models.list();
};

getModels().then((models) => {
  // console.log(models);
});


grog cloud
Playground
API Keys
Dashboard
Documentation
Upgrade

Steven 
Personal
Docs
API Reference
Get Started
Overview
Quickstart
OpenAI Compatibility
Models
Rate Limits
Features
Text
Speech to Text
Text to Speech
Reasoning
Vision
Agentic Tooling
Advanced Features
Batch Processing
Flex Processing
Content Moderation
Prefilling
Tool Use
Developer Resources
Groq Libraries
Groq Badge
Examples
Applications Showcase
Resources
Prompting Guide
Integrations
Integrations Catalog
Support & Guidelines
Errors
Changelog
Policies & Notices
Quickstart
Get up and running with the Groq API in a few minutes.

Create an API Key
Please visit here to create an API Key.

Set up your API Key (recommended)
Configure your API key as an environment variable. This approach streamlines your API usage by eliminating the need to include your API key in each request. Moreover, it enhances security by minimizing the risk of inadvertently including your API key in your codebase.

In your terminal of choice:

export GROQ_API_KEY=<your-api-key-here>
Requesting your first chat completion
curl
JavaScript
Python
JSON
Install the Groq JavaScript library:

npm install --save groq-sdk
Performing a Chat Completion:

import Groq from "groq-sdk";

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

export async function main() {
  const chatCompletion = await getGroqChatCompletion();
  // Print the completion returned by the LLM.
  console.log(chatCompletion.choices[0]?.message?.content || "");
}

export async function getGroqChatCompletion() {
  return groq.chat.completions.create({
    messages: [
      {
        role: "user",
        content: "Explain the importance of fast language models",
      },
    ],
    model: "llama-3.3-70b-versatile",
  });
}
Now that you have successfully received a chat completion, you can try out the other endpoints in the API.

Next Steps
Check out the Playground to try out the Groq API in your browser
Join our GroqCloud developer community on Discord
Add a how-to on your project to the Groq API Cookbook


grog cloud
Playground
API Keys
Dashboard
Documentation
Upgrade

Steven 
Personal
Docs
API Reference
Get Started
Overview
Quickstart
OpenAI Compatibility
Models
Rate Limits
Features
Text
Speech to Text
Text to Speech
Reasoning
Vision
Agentic Tooling
Advanced Features
Batch Processing
Flex Processing
Content Moderation
Prefilling
Tool Use
Developer Resources
Groq Libraries
Groq Badge
Examples
Applications Showcase
Resources
Prompting Guide
Integrations
Integrations Catalog
Support & Guidelines
Errors
Changelog
Policies & Notices
Chat Completion Models
The Groq Chat Completions API processes a series of messages and generates output responses. These models can perform multi-turn discussions or tasks that require only one interaction.


For details about the parameters, visit the reference page.

JSON mode
JSON mode is a feature that guarantees all chat completions are valid JSON.

Usage:

Set "response_format": {"type": "json_object"} in your chat completion request
Add a description of the desired JSON structure within the system prompt (see below for example system prompts)
Recommendations for best JSON results:

Llama performs best at generating JSON, followed by Gemma, then Mixtral
Use pretty-printed JSON instead of compact JSON
Keep prompts concise
JSON mode Limitations:

Does not support streaming
Does not support stop sequences
Error Code:

Groq will return a 400 error with an error code of json_validate_failed if JSON generation fails.
Example system prompts:



You are a legal advisor who summarizes documents in JSON


You are a data analyst API capable of sentiment analysis that responds in JSON.  The JSON schema should include
{
  "sentiment_analysis": {
    "sentiment": "string (positive, negative, neutral)",
    "confidence_score": "number (0-1)"
    # Include additional fields as required
  }
}
Generating Chat Completions with Groq SDK
Code Overview
Python
JavaScript

npm install --save groq-sdk

Performing a basic Chat Completion

import Groq from "groq-sdk";

const groq = new Groq();

export async function main() {
  const chatCompletion = await getGroqChatCompletion();
  // Print the completion returned by the LLM.
  console.log(chatCompletion.choices[0]?.message?.content || "");
}

export const getGroqChatCompletion = async () => {
  return groq.chat.completions.create({
    //
    // Required parameters
    //
    messages: [
      // Set an optional system message. This sets the behavior of the
      // assistant and can be used to provide specific instructions for
      // how it should behave throughout the conversation.
      {
        role: "system",
        content: "you are a helpful assistant.",
      },
      // Set a user message for the assistant to respond to.
      {
        role: "user",
        content: "Explain the importance of fast language models",
      },
    ],

    // The language model which will generate the completion.
    model: "llama-3.3-70b-versatile",

    //
    // Optional parameters
    //

    // Controls randomness: lowering results in less random completions.
    // As the temperature approaches zero, the model will become deterministic
    // and repetitive.
    temperature: 0.5,

    // The maximum number of tokens to generate. Requests can use up to
    // 2048 tokens shared between prompt and completion.
    max_completion_tokens: 1024,

    // Controls diversity via nucleus sampling: 0.5 means half of all
    // likelihood-weighted options are considered.
    top_p: 1,

    // A stop sequence is a predefined or user-specified text string that
    // signals an AI to stop generating content, ensuring its responses
    // remain focused and concise. Examples include punctuation marks and
    // markers like "[end]".
    stop: null,

    // If set, partial message deltas will be sent.
    stream: false,
  });
};

main();

Streaming a Chat Completion
To stream a completion, simply set the parameter stream=True. Then the completion function will return an iterator of completion deltas rather than a single, full completion.



import Groq from "groq-sdk";

const groq = new Groq();

export async function main() {
  const stream = await getGroqChatStream();
  for await (const chunk of stream) {
    // Print the completion returned by the LLM.
    process.stdout.write(chunk.choices[0]?.delta?.content || "");
  }
}

export async function getGroqChatStream() {
  return groq.chat.completions.create({
    //
    // Required parameters
    //
    messages: [
      // Set an optional system message. This sets the behavior of the
      // assistant and can be used to provide specific instructions for
      // how it should behave throughout the conversation.
      {
        role: "system",
        content: "you are a helpful assistant.",
      },
      // Set a user message for the assistant to respond to.
      {
        role: "user",
        content: "Explain the importance of fast language models",
      },
    ],

    // The language model which will generate the completion.
    model: "llama-3.3-70b-versatile",

    //
    // Optional parameters
    //

    // Controls randomness: lowering results in less random completions.
    // As the temperature approaches zero, the model will become deterministic
    // and repetitive.
    temperature: 0.5,

    // The maximum number of tokens to generate. Requests can use up to
    // 2048 tokens shared between prompt and completion.
    max_completion_tokens: 1024,

    // Controls diversity via nucleus sampling: 0.5 means half of all
    // likelihood-weighted options are considered.
    top_p: 1,

    // A stop sequence is a predefined or user-specified text string that
    // signals an AI to stop generating content, ensuring its responses
    // remain focused and concise. Examples include punctuation marks and
    // markers like "[end]".
    stop: null,

    // If set, partial message deltas will be sent.
    stream: true,
  });
}

main();
Streaming a chat completion with a stop sequence

import Groq from "groq-sdk";

const groq = new Groq();

export async function main() {
  const stream = await getGroqChatStream();
  for await (const chunk of stream) {
    // Print the completion returned by the LLM.
    process.stdout.write(chunk.choices[0]?.delta?.content || "");
  }
}

export async function getGroqChatStream() {
  return groq.chat.completions.create({
    //
    // Required parameters
    //
    messages: [
      // Set an optional system message. This sets the behavior of the
      // assistant and can be used to provide specific instructions for
      // how it should behave throughout the conversation.
      {
        role: "system",
        content: "you are a helpful assistant.",
      },
      // Set a user message for the assistant to respond to.
      {
        role: "user",
        content:
          "Start at 1 and count to 10.  Separate each number with a comma and a space",
      },
    ],

    // The language model which will generate the completion.
    model: "llama-3.3-70b-versatile",

    //
    // Optional parameters
    //

    // Controls randomness: lowering results in less random completions.
    // As the temperature approaches zero, the model will become deterministic
    // and repetitive.
    temperature: 0.5,

    // The maximum number of tokens to generate. Requests can use up to
    // 2048 tokens shared between prompt and completion.
    max_completion_tokens: 1024,

    // Controls diversity via nucleus sampling: 0.5 means half of all
    // likelihood-weighted options are considered.
    top_p: 1,

    // A stop sequence is a predefined or user-specified text string that
    // signals an AI to stop generating content, ensuring its responses
    // remain focused and concise. Examples include punctuation marks and
    // markers like "[end]".
    //
    // For this example, we will use ", 6" so that the llm stops counting at 5.
    // If multiple stop values are needed, an array of string may be passed,
    // stop: [", 6", ", six", ", Six"]
    stop: ", 6",

    // If set, partial message deltas will be sent.
    stream: true,
  });
}

main();
JSON Mode

import Groq from "groq-sdk";
const groq = new Groq();

// Define the JSON schema for recipe objects
// This is the schema that the model will use to generate the JSON object, 
// which will be parsed into the Recipe class.
const schema = {
  $defs: {
    Ingredient: {
      properties: {
        name: { title: "Name", type: "string" },
        quantity: { title: "Quantity", type: "string" },
        quantity_unit: {
          anyOf: [{ type: "string" }, { type: "null" }],
          title: "Quantity Unit",
        },
      },
      required: ["name", "quantity", "quantity_unit"],
      title: "Ingredient",
      type: "object",
    },
  },
  properties: {
    recipe_name: { title: "Recipe Name", type: "string" },
    ingredients: {
      items: { $ref: "#/$defs/Ingredient" },
      title: "Ingredients",
      type: "array",
    },
    directions: {
      items: { type: "string" },
      title: "Directions",
      type: "array",
    },
  },
  required: ["recipe_name", "ingredients", "directions"],
  title: "Recipe",
  type: "object",
};

// Ingredient class representing a single recipe ingredient
class Ingredient {
  constructor(name, quantity, quantity_unit) {
    this.name = name;
    this.quantity = quantity;
    this.quantity_unit = quantity_unit || null;
  }
}

// Recipe class representing a complete recipe
class Recipe {
  constructor(recipe_name, ingredients, directions) {
    this.recipe_name = recipe_name;
    this.ingredients = ingredients;
    this.directions = directions;
  }
}

// Generates a recipe based on the recipe name
export async function getRecipe(recipe_name) {
  // Pretty printing improves completion results
  const jsonSchema = JSON.stringify(schema, null, 4);
  const chat_completion = await groq.chat.completions.create({
    messages: [
      {
        role: "system",
        content: `You are a recipe database that outputs recipes in JSON.\n'The JSON object must use the schema: ${jsonSchema}`,
      },
      {
        role: "user",
        content: `Fetch a recipe for ${recipe_name}`,
      },
    ],
    model: "llama-3.3-70b-versatile",
    temperature: 0,
    stream: false,
    response_format: { type: "json_object" },
  });

  const recipeJson = JSON.parse(chat_completion.choices[0].message.content);

  // Map the JSON ingredients to the Ingredient class
  const ingredients = recipeJson.ingredients.map((ingredient) => {
    return new Ingredient(ingredient.name, ingredient.quantity, ingredient.quantity_unit);
  });

  // Return the recipe object
  return new Recipe(recipeJson.recipe_name, ingredients, recipeJson.directions);
}

// Prints a recipe to the console with nice formatting
function printRecipe(recipe) {
  console.log("Recipe:", recipe.recipe_name);
  console.log();

  console.log("Ingredients:");
  recipe.ingredients.forEach((ingredient) => {
    console.log(
      `- ${ingredient.name}: ${ingredient.quantity} ${
        ingredient.quantity_unit || ""
      }`,
    );
  });
  console.log();

  console.log("Directions:");
  recipe.directions.forEach((direction, step) => {
    console.log(`${step + 1}. ${direction}`);
  });
}

// Main function that generates and prints a recipe
export async function main() {
  const recipe = await getRecipe("apple pie");
  printRecipe(recipe);
}

main();



grog cloud
Playground
API Keys
Dashboard
Documentation
Upgrade

Steven 
Personal
Docs
API Reference
Get Started
Overview
Quickstart
OpenAI Compatibility
Models
Rate Limits
Features
Text
Speech to Text
Text to Speech
Reasoning
Vision
Agentic Tooling
Advanced Features
Batch Processing
Flex Processing
Content Moderation
Prefilling
Tool Use
Developer Resources
Groq Libraries
Groq Badge
Examples
Applications Showcase
Resources
Prompting Guide
Integrations
Integrations Catalog
Support & Guidelines
Errors
Changelog
Policies & Notices
Speech to Text
Groq API is the fastest speech-to-text solution available, offering OpenAI-compatible endpoints that enable near-instant transcriptions and translations. With Groq API, you can integrate high-quality audio processing into your applications at speeds that rival human interaction.

API Endpoints
We support two endpoints:

Endpoint	Usage	API Endpoint
Transcriptions	Convert audio to text	https://api.groq.com/openai/v1/audio/transcriptions
Translations	Translate audio to English text	https://api.groq.com/openai/v1/audio/translations
Supported Models
Model ID	Model	Supported Language(s)	Description
whisper-large-v3-turbo

Whisper Large V3 Turbo	Multilingual	A fine-tuned version of a pruned Whisper Large V3 designed for fast, multilingual transcription tasks.
distil-whisper-large-v3-en

Distil-Whisper English	English-only	A distilled, or compressed, version of OpenAI's Whisper model, designed to provide faster, lower cost English speech recognition while maintaining comparable accuracy.
whisper-large-v3

Whisper large-v3	Multilingual	Provides state-of-the-art performance with high accuracy for multilingual transcription and translation tasks.
Which Whisper Model Should You Use?
Having more choices is great, but let's try to avoid decision paralysis by breaking down the tradeoffs between models to find the one most suitable for your applications:

If your application is error-sensitive and requires multilingual support, use 
whisper-large-v3

.
If your application is less sensitive to errors and requires English only, use 
distil-whisper-large-v3-en

.
If your application requires multilingual support and you need the best price for performance, use 
whisper-large-v3-turbo

.
The following table breaks down the metrics for each model.

Model	Cost Per Hour	Language Support	Transcription Support	Translation Support	Real-time Speed Factor	Word Error Rate
whisper-large-v3

$0.111	Multilingual	Yes	Yes	189	10.3%
whisper-large-v3-turbo

$0.04	Multilingual	Yes	No	216	12%
distil-whisper-large-v3-en

$0.02	English only	Yes	No	250	13%
Working with Audio Files
Audio File Limitations
Max File Size
40 MB (free tier), 100MB (dev tier)

Max Attachment File Size
25 MB. If you need to process larger files, use the url parameter to specify a url to the file instead.

Minimum File Length
0.01 seconds

Minimum Billed Length
10 seconds. If you submit a request less than this, you will still be billed for 10 seconds.

Supported File Types
Either a URL or a direct file upload for flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm

Single Audio Track
Only the first track will be transcribed for files with multiple audio tracks. (e.g. dubbed video)

Supported Response Formats
json, verbose_json, text

Supported Timestamp Granularities
segment, word

Audio Request Examples
I am on free tier and want to transcribe an audio file:

Use the file parameter to add a local file up to 25 MB.
Use the url parameter to add a url to a file up to 40 MB.
I am on dev tier and want to transcribe an audio file:

Use the file parameter to add a local file up to 25 MB.
Use the url parameter to add a url to a file up to 100 MB.
If audio files exceed these limits, you may receive a 413 error.

Audio Preprocessing
Our speech-to-text models will downsample audio to 16KHz mono before transcribing, which is optimal for speech recognition. This preprocessing can be performed client-side if your original file is extremely large and you want to make it smaller without a loss in quality (without chunking, Groq API speech-to-text endpoints accept up to 40MB for free tier and 100MB for dev tier). We recommend FLAC for lossless compression.

The following ffmpeg command can be used to reduce file size:


ffmpeg \
  -i <your file> \
  -ar 16000 \
  -ac 1 \
  -map 0:a \
  -c:a flac \
  <output file name>.flac
Working with Larger Audio Files
For audio files that exceed our size limits or require more precise control over transcription, we recommend implementing audio chunking. This process involves:

Breaking the audio into smaller, overlapping segments
Processing each segment independently
Combining the results while handling overlapping
To learn more about this process and get code for your own implementation, see the complete audio chunking tutorial in our Groq API Cookbook. 

Using the API
The following are request parameters you can use in your transcription and translation requests:

Parameter	Type	Default	Description
file	string	Required unless using url instead	The audio file object for direct upload to translate/transcribe.
url	string	Required unless using file instead	The audio URL to translate/transcribe (supports Base64URL).
language	string	Optional	The language of the input audio. Supplying the input language in ISO-639-1 (i.e. en, tr`) format will improve accuracy and latency.

The translations endpoint only supports 'en' as a parameter option.
model	string	Required	ID of the model to use.
prompt	string	Optional	Prompt to guide the model's style or specify how to spell unfamiliar words. (limited to 224 tokens)
response_format	string	json	Define the output response format.

Set to verbose_json to receive timestamps for audio segments.

Set to text to return a text response.
temperature	float	0	The temperature between 0 and 1. For translations and transcriptions, we recommend the default value of 0.
timestamp_granularities[]	array	segment	The timestamp granularities to populate for this transcription. response_format must be set verbose_json to use timestamp granularities.

Either or both of word and segment are supported.

segment returns full metadata and word returns only word, start, and end timestamps. To get both word-level timestamps and full segment metadata, include both values in the array.
Example Usage of Transcription Endpoint
The transcription endpoint allows you to transcribe spoken words in audio or video files.

Python
JavaScript
curl
The Groq SDK package can be installed using the following command:


npm install --save groq-sdk
The following code snippet demonstrates how to use Groq API to transcribe an audio file in JavaScript:


import fs from "fs";
import Groq from "groq-sdk";

// Initialize the Groq client
const groq = new Groq();

async function main() {
  // Create a transcription job
  const transcription = await groq.audio.transcriptions.create({
    file: fs.createReadStream("YOUR_AUDIO.wav"), // Required path to audio file - replace with your audio file!
    model: "whisper-large-v3-turbo", // Required model to use for transcription
    prompt: "Specify context or spelling", // Optional
    response_format: "verbose_json", // Optional
    timestamp_granularities: ["word", "segment"], // Optional (must set response_format to "json" to use and can specify "word", "segment" (default), or both)
    language: "en", // Optional
    temperature: 0.0, // Optional
  });
  // To print only the transcription text, you'd use console.log(transcription.text); (here we're printing the entire transcription object to access timestamps)
  console.log(JSON.stringify(transcription, null, 2));
}
main();
Example Usage of Translation Endpoint
The translation endpoint allows you to translate spoken words in audio or video files to English.

Python
JavaScript
curl
The Groq SDK package can be installed using the following command:


npm install --save groq-sdk
The following code snippet demonstrates how to use Groq API to translate an audio file in JavaScript:


import fs from "fs";
import Groq from "groq-sdk";

// Initialize the Groq client
const groq = new Groq();
async function main() {
  // Create a translation job
  const translation = await groq.audio.translations.create({
    file: fs.createReadStream("sample_audio.m4a"), // Required path to audio file - replace with your audio file!
    model: "whisper-large-v3", // Required model to use for translation
    prompt: "Specify context or spelling", // Optional
    language: "en", // Optional ('en' only)
    response_format: "json", // Optional
    temperature: 0.0, // Optional
  });
  // Log the transcribed text
  console.log(translation.text);
}
main();
Understanding Metadata Fields
When working with Groq API, setting response_format to verbose_json outputs each segment of transcribed text with valuable metadata that helps us understand the quality and characteristics of our transcription, including avg_logprob, compression_ratio, and no_speech_prob.

This information can help us with debugging any transcription issues. Let's examine what this metadata tells us using a real example:


{
  "id": 8,
  "seek": 3000,
  "start": 43.92,
  "end": 50.16,
  "text": " document that the functional specification that you started to read through that isn't just the",
  "tokens": [51061, 4166, 300, 264, 11745, 31256],
  "temperature": 0,
  "avg_logprob": -0.097569615,
  "compression_ratio": 1.6637554,
  "no_speech_prob": 0.012814695
}
As shown in the above example, we receive timing information as well as quality indicators. Let's gain a better understanding of what each field means:

id:8: The 9th segment in the transcription (counting begins at 0)
seek: Indicates where in the audio file this segment begins (3000 in this case)
start and end timestamps: Tell us exactly when this segment occurs in the audio (43.92 to 50.16 seconds in our example)
avg_logprob (Average Log Probability): -0.097569615 in our example indicates very high confidence. Values closer to 0 suggest better confidence, while more negative values (like -0.5 or lower) might indicate transcription issues.
no_speech_prob (No Speech Probability): 0.0.012814695 is very low, suggesting this is definitely speech. Higher values (closer to 1) would indicate potential silence or non-speech audio.
compression_ratio: 1.6637554 is a healthy value, indicating normal speech patterns. Unusual values (very high or low) might suggest issues with speech clarity or word boundaries.
Using Metadata for Debugging
When troubleshooting transcription issues, look for these patterns:

Low Confidence Sections: If avg_logprob drops significantly (becomes more negative), check for background noise, multiple speakers talking simultaneously, unclear pronunciation, and strong accents. Consider cleaning up the audio in these sections or adjusting chunk sizes around problematic chunk boundaries.
Non-Speech Detection: High no_speech_prob values might indicate silence periods that could be trimmed, background music or noise, or non-verbal sounds being misinterpreted as speech. Consider noise reduction when preprocessing.
Unusual Speech Patterns: Unexpected compression_ratio values can reveal stuttering or word repetition, speaker talking unusually fast or slow, or audio quality issues affecting word separation.
Quality Thresholds and Regular Monitoring
We recommend setting acceptable ranges for each metadata value we reviewed above and flagging segments that fall outside these ranges to be able to identify and adjust preprocessing or chunking strategies for flagged sections.

By understanding and monitoring these metadata values, you can significantly improve your transcription quality and quickly identify potential issues in your audio processing pipeline.

Prompting Guidelines
The prompt parameter (max 224 tokens) helps provide context and maintain a consistent output style. Unlike chat completion prompts, these prompts only guide style and context, not specific actions.

Best Practices
Provide relevant context about the audio content, such as the type of conversation, topic, or speakers involved.
Use the same language as the language of the audio file.
Steer the model's output by denoting proper spellings or emulate a specific writing style or tone.
Keep the prompt concise and focused on stylistic guidance.
We can't wait to see what you build! 🚀



grog cloud
Playground
API Keys
Dashboard
Documentation
Upgrade

Steven 
Personal
Docs
API Reference
Get Started
Overview
Quickstart
OpenAI Compatibility
Models
Rate Limits
Features
Text
Speech to Text
Text to Speech
Reasoning
Vision
Agentic Tooling
Advanced Features
Batch Processing
Flex Processing
Content Moderation
Prefilling
Tool Use
Developer Resources
Groq Libraries
Groq Badge
Examples
Applications Showcase
Resources
Prompting Guide
Integrations
Integrations Catalog
Support & Guidelines
Errors
Changelog
Policies & Notices
Text to Speech
Learn how to instantly generate lifelike audio from text.

Overview
The Groq API speech endpoint provides fast text-to-speech (TTS), enabling you to convert text to spoken audio in seconds with our available TTS models.

With support for 23 voices, 19 in English and 4 in Arabic, you can instantly create life-like audio content for customer support agents, characters for game development, and more.

API Endpoint
Endpoint	Usage	API Endpoint
Speech	Convert text to audio	https://api.groq.com/openai/v1/audio/speech
Supported Models
Model ID	Model Card	Supported Language(s)	Description
playai-tts	Card 	English	High-quality TTS model for English speech generation.
playai-tts-arabic	Card 	Arabic	High-quality TTS model for Arabic speech generation.
Working with Speech
Quick Start
The speech endpoint takes four key inputs:

model: playai-tts or playai-tts-arabic
input: the text to generate audio from
voice: the desired voice for output
response format: defaults to "wav"
Python
JavaScript
curl
The Groq SDK package can be installed using the following command:


npm install --save groq-sdk
The following is an example of a request using playai-tts. To use the Arabic model, use the playai-tts-arabic model ID and an Arabic prompt:


import fs from "fs";
import path from "path";
import Groq from 'groq-sdk';

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY
});

const speechFilePath = "speech.wav";
const model = "playai-tts";
const voice = "Fritz-PlayAI";
const text = "I love building and shipping new features for our users!";
const responseFormat = "wav";

async function main() {
  const response = await groq.audio.speech.create({
    model: model,
    voice: voice,
    input: text,
    response_format: responseFormat
  });
  
  const buffer = Buffer.from(await response.arrayBuffer());
  await fs.promises.writeFile(speechFilePath, buffer);
}

main();
Parameters
Parameter	Type	Required	Value	Description
model	string	Yes	playai-tts
playai-tts-arabic	Model ID to use for TTS.
input	string	Yes	-	User input text to be converted to speech. Maximum length is 10K characters.
voice	string	Yes	See available English and Arabic voices.	The voice to use for audio generation. There are currently 26 English options for playai-tts and 4 Arabic options for playai-tts-arabic.
response_format	string	Optional	"wav"	Format of the response audio file. Defaults to currently supported "wav".
Available English Voices
The playai-tts model currently supports 19 English voices that you can pass into the voice parameter (Arista-PlayAI, Atlas-PlayAI, Basil-PlayAI, Briggs-PlayAI, Calum-PlayAI, Celeste-PlayAI, Cheyenne-PlayAI, Chip-PlayAI, Cillian-PlayAI, Deedee-PlayAI, Fritz-PlayAI, Gail-PlayAI, Indigo-PlayAI, Mamaw-PlayAI, Mason-PlayAI, Mikail-PlayAI, Mitch-PlayAI, Quinn-PlayAI, Thunder-PlayAI).

Experiment to find the voice you need for your application:

Arista-PlayAI


0:00
0:03

Atlas-PlayAI


0:00
0:04

Basil-PlayAI


0:00
0:03

Briggs-PlayAI


0:00
0:03

Calum-PlayAI


0:00
0:03

Celeste-PlayAI


0:00
0:03

Cheyenne-PlayAI


0:00
0:03

Chip-PlayAI


0:00
0:03

Cillian-PlayAI


0:00
0:02

Deedee-PlayAI


0:00
0:04

Fritz-PlayAI


0:00
0:03

Gail-PlayAI


0:00
0:03

Indigo-PlayAI


0:00
0:03

Mamaw-PlayAI


0:00
0:03

Mason-PlayAI


0:00
0:03

Mikail-PlayAI


0:00
0:03

Mitch-PlayAI


0:00
0:03

Quinn-PlayAI


0:00
0:03

Thunder-PlayAI


0:00
0:03

Available Arabic Voices
The playai-tts-arabic model currently supports 4 Arabic voices that you can pass into the voice parameter (Ahmad-PlayAI, Amira-PlayAI, Khalid-PlayAI, Nasser-PlayAI).

Experiment to find the voice you need for your application:

Ahmad-PlayAI


0:00
0:03

Amira-PlayAI


0:00
0:03

Khalid-PlayAI


0:00
0:02

Nasser-PlayAI


0:00
0:03


grog cloud
Playground
API Keys
Dashboard
Documentation
Upgrade

Steven 
Personal
Docs
API Reference
Get Started
Overview
Quickstart
OpenAI Compatibility
Models
Rate Limits
Features
Text
Speech to Text
Text to Speech
Reasoning
Vision
Agentic Tooling
Advanced Features
Batch Processing
Flex Processing
Content Moderation
Prefilling
Tool Use
Developer Resources
Groq Libraries
Groq Badge
Examples
Applications Showcase
Resources
Prompting Guide
Integrations
Integrations Catalog
Support & Guidelines
Errors
Changelog
Policies & Notices
Reasoning
Reasoning models excel at complex problem-solving tasks that require step-by-step analysis, logical deduction, and structured thinking and solution validation. With Groq inference speed, these types of models can deliver instant reasoning capabilities critical for real-time applications.

Why Speed Matters for Reasoning
Reasoning models are capable of complex decision making with explicit reasoning chains that are part of the token output and used for decision-making, which make low-latency and fast inference essential. Complex problems often require multiple chains of reasoning tokens where each step build on previous results. Low latency compounds benefits across reasoning chains and shaves off minutes of reasoning to a response in seconds.

Supported Model
Model ID	Model
qwen-qwq-32b

Qwen QwQ 32B
deepseek-r1-distill-llama-70b

DeepSeek R1 Distil Llama 70B
Reasoning Format
Groq API supports explicit reasoning formats through the reasoning_format parameter, giving you fine-grained control over how the model's reasoning process is presented. This is particularly valuable for valid JSON outputs, debugging, and understanding the model's decision-making process.

Note: The format defaults to raw or parsed when JSON mode or tool use are enabled as those modes do not support raw. If reasoning is explicitly set to raw with JSON mode or tool use enabled, we will return a 400 error.

Options for Reasoning Format
reasoning_format Options	Description
parsed	Separates reasoning into a dedicated field while keeping the response concise.
raw	Includes reasoning within think tags in the content.
hidden	Returns only the final answer.
Quick Start
Python
JavaScript
curl

import Groq from 'groq-sdk';

const client = new Groq();
const completion = await client.chat.completions.create({
    model: "deepseek-r1-distill-llama-70b",
    messages: [
        {
            role: "user",
            content: "How many r's are in the word strawberry?"
        }
    ],
    temperature: 0.6,
    max_completion_tokens: 1024,
    top_p: 0.95,
    stream: true,
    reasoning_format: "raw"
});

for await (const chunk of completion) {
    process.stdout.write(chunk.choices[0].delta.content || "");
}
Quick Start with Tool use

curl https://api.groq.com//openai/v1/chat/completions -s \
  -H "authorization: bearer $GROQ_API_KEY" \
  -d '{
    "model": "deepseek-r1-distill-llama-70b",
    "messages": [
        {
            "role": "user",
            "content": "What is the weather like in Paris today?"
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current temperature for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Bogotá, Colombia"
                        }
                    },
                    "required": [
                        "location"
                    ],
                    "additionalProperties": false
                },
                "strict": true
            }
        }
    ]}'
Recommended Configuration Parameters
Parameter	Default	Range	Description
messages	-	-	Array of message objects. Important: Avoid system prompts - include all instructions in the user message!
temperature	0.6	0.0 - 2.0	Controls randomness in responses. Lower values make responses more deterministic. Recommended range: 0.5-0.7 to prevent repetitions or incoherent outputs
max_completion_tokens	1024	-	Maximum length of model's response. Default may be too low for complex reasoning - consider increasing for detailed step-by-step solutions
top_p	0.95	0.0 - 1.0	Controls diversity of token selection
stream	false	boolean	Enables response streaming. Recommended for interactive reasoning tasks
stop	null	string/array	Custom stop sequences
seed	null	integer	Set for reproducible results. Important for benchmarking - run multiple tests with different seeds
response_format	{type: "text"}	{type: "json_object"} or {type: "text"}	Set to json_object type for structured output.
reasoning_format	raw	"parsed", "raw", "hidden"	Controls how model reasoning is presented in the response. Must be set to either parsed or hidden when using tool calling or JSON mode.
Optimizing Performance
Temperature and Token Management
The model performs best with temperature settings between 0.5-0.7, with lower values (closer to 0.5) producing more consistent mathematical proofs and higher values allowing for more creative problem-solving approaches. Monitor and adjust your token usage based on the complexity of your reasoning tasks - while the default max_completion_tokens is 1024, complex proofs may require higher limits.

Prompt Engineering
To ensure accurate, step-by-step reasoning while maintaining high performance:

DeepSeek-R1 works best when all instructions are included directly in user messages rather than system prompts.
Structure your prompts to request explicit validation steps and intermediate calculations.
Avoid few-shot prompting and go for zero-shot prompting only.


grog cloud
Playground
API Keys
Dashboard
Documentation
Upgrade

Steven 
Personal
Docs
API Reference
Get Started
Overview
Quickstart
OpenAI Compatibility
Models
Rate Limits
Features
Text
Speech to Text
Text to Speech
Reasoning
Vision
Agentic Tooling
Advanced Features
Batch Processing
Flex Processing
Content Moderation
Prefilling
Tool Use
Developer Resources
Groq Libraries
Groq Badge
Examples
Applications Showcase
Resources
Prompting Guide
Integrations
Integrations Catalog
Support & Guidelines
Errors
Changelog
Policies & Notices
Vision
Groq API offers fast inference and low latency for multimodal models with vision capabilities for understanding and interpreting visual data from images. By analyzing the content of an image, multimodal models can generate human-readable text for providing insights about given visual data.

Supported Models
Groq API supports powerful multimodal models that can be easily integrated into your applications to provide fast and accurate image processing for tasks such as visual question answering, caption generation, and Optical Character Recognition (OCR).

Llama 4 Scout
Llama 4 Maverick
meta-llama/llama-4-scout-17b-16e-instruct
Model ID
meta-llama/llama-4-scout-17b-16e-instruct

Description
A powerful multimodal model capable of processing both text and image inputs that supports multilingual, multi-turn conversations, tool use, and JSON mode.

Context Window
10M tokens (limited to 128K in preview)

Preview Model
Currently in preview and should be used for experimentation.

Image Size Limit
Maximum allowed size for a request containing an image URL as input is 20MB. Requests larger than this limit will return a 400 error.

Image Resolution Limit
Maximum allowed resolution for a request containing images is 33 megapixels (33177600 total pixels) per image. Images larger than this limit will return a 400 error.

Request Size Limit (Base64 Encoded Images)
Maximum allowed size for a request containing a base64 encoded image is 4MB. Requests larger than this limit will return a 413 error.

Images per Request
You can process as many images as you want in a single request, but we recommend a maximum of 5 for highest quality and accuracy in responses as included in guidelines by Meta.

How to Use Vision
Use Groq API vision features via:

GroqCloud Console Playground: Use Llama 4 Scout or Llama 4 Maverick as the model and upload your image.
Groq API Request: Call the chat.completions API endpoint and set the model to 
meta-llama/llama-4-scout-17b-16e-instruct

 or 
meta-llama/llama-4-maverick-17b-128e-instruct

. See code examples below.

How to Pass Images from URLs as Input
The following are code examples for passing your image to the model via a URL:

curl
JavaScript
Python
JSON

import { Groq } from 'groq-sdk';

const groq = new Groq();
async function main() {
  const chatCompletion = await groq.chat.completions.create({
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What's in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://upload.wikimedia.org/wikipedia/commons/f/f2/LPU-v1-die.jpg"
            }
          }
        ]
      }
    ],
    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    "temperature": 1,
    "max_completion_tokens": 1024,
    "top_p": 1,
    "stream": false,
    "stop": null
  });

   console.log(chatCompletion.choices[0].message.content);
}

main();

How to Pass Locally Saved Images as Input
To pass locally saved images, we'll need to first encode our image to a base64 format string before passing it as the image_url in our API request as follows:



from groq import Groq
import base64
import os

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "sf.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    model="meta-llama/llama-4-scout-17b-16e-instruct",
)

print(chat_completion.choices[0].message.content)

Tool Use with Images
The meta-llama/llama-4-scout-17b-16e-instruct, meta-llama/llama-4-maverick-17b-128e-instruct models support tool use! The following cURL example defines a get_current_weather tool that the model can leverage to answer a user query that contains a question about the weather along with an image of a location that the model can infer location (i.e. New York City) from:



curl https://api.groq.com/openai/v1/chat/completions -s \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $GROQ_API_KEY" \
-d '{
"model": "meta-llama/llama-4-scout-17b-16e-instruct",
"messages": [
{
    "role": "user",
    "content": [{"type": "text", "text": "Whats the weather like in this state?"}, {"type": "image_url", "image_url": { "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"}}]
}
],
"tools": [
{
    "type": "function",
    "function": {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
        "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
        },
        "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
        }
        },
        "required": ["location"]
    }
    }
}
],
"tool_choice": "auto"
}' | jq '.choices[0].message.tool_calls'

The following is the output from our example above that shows how our model inferred the state as New York from the given image and called our example function:



[
  {
    "id": "call_q0wg",
    "function": {
      "arguments": "{\"location\": \"New York, NY\",\"unit\": \"fahrenheit\"}",
      "name": "get_current_weather"
    },
    "type": "function"
  }
]

JSON Mode with Images
The meta-llama/llama-4-scout-17b-16e-instruct and meta-llama/llama-4-maverick-17b-128e-instruct models support JSON mode! The following Python example queries the model with an image and text (i.e. "Please pull out relevant information as a JSON object.") with response_format set for JSON mode:



from groq import Groq
import os

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

completion = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "List what you observe in this photo in JSON format."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/d/da/SF_From_Marin_Highlands3.jpg"
                    }
                }
            ]
        }
    ],
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    stream=False,
    response_format={"type": "json_object"},
    stop=None,
)

print(completion.choices[0].message)

Multi-turn Conversations with Images
The meta-llama/llama-4-scout-17b-16e-instruct and meta-llama/llama-4-maverick-17b-128e-instruct models support multi-turn conversations! The following Python example shows a multi-turn user conversation about an image:



from groq import Groq
import os

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

completion = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/d/da/SF_From_Marin_Highlands3.jpg"
                    }
                }
            ]
        },
        {
            "role": "user",
            "content": "Tell me more about the area."
        }
    ],
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    stream=False,
    stop=None,
)

print(completion.choices[0].message)

Venture Deeper into Vision
Use Cases to Explore
Vision models can be used in a wide range of applications. Here are some ideas:

Accessibility Applications: Develop an application that generates audio descriptions for images by using a vision model to generate text descriptions for images, which can then be converted to audio with one of our audio endpoints.
E-commerce Product Description Generation: Create an application that generates product descriptions for e-commerce websites.
Multilingual Image Analysis: Create applications that can describe images in multiple languages.
Multi-turn Visual Conversations: Develop interactive applications that allow users to have extended conversations about images.
These are just a few ideas to get you started. The possibilities are endless, and we're excited to see what you create with vision models powered by Groq for low latency and fast inference!


Next Steps
Check out our Groq API Cookbook repository on GitHub (and give us a ⭐) for practical examples and tutorials:

Image Moderation
Multimodal Image Processing (Tool Use, JSON Mode)

We're always looking for contributions. If you have any cool tutorials or guides to share, submit a pull request for review to help our open-source community!



grog cloud
Playground
API Keys
Dashboard
Documentation
Upgrade

Steven 
Personal
Docs
API Reference
Get Started
Overview
Quickstart
OpenAI Compatibility
Models
Rate Limits
Features
Text
Speech to Text
Text to Speech
Reasoning
Vision
Agentic Tooling
Advanced Features
Batch Processing
Flex Processing
Content Moderation
Prefilling
Tool Use
Developer Resources
Groq Libraries
Groq Badge
Examples
Applications Showcase
Resources
Prompting Guide
Integrations
Integrations Catalog
Support & Guidelines
Errors
Changelog
Policies & Notices
Agentic Tooling
While LLMs excel at generating text, compound-beta takes the next step. It's an advanced AI system that is designed to solve problems by taking action and intelligently uses external tools - starting with web search and code execution - alongside the powerful Llama 4 models and Llama 3.3 70b model. This allows it access to real-time information and interaction with external environments, providing more accurate, up-to-date, and capable responses than an LLM alone.

Available Agentic Tools
There are two agentic tool systems available:

compound-beta: supports multiple tool calls per request. This system is great for use cases that require multiple web searches or code executions per request.
compound-beta-mini: supports a single tool call per request. This system is great for use cases that require a single web search or code execution per request. compound-beta-mini has an average of 3x lower latency than compound-beta.
Both systems support the following tools:

Web Search via Tavily
Code Execution via E2B (only Python is currently supported)

Custom user-provided tools are not supported at this time.

Usage
To use agentic tools, change the model parameter to either compound-beta or compound-beta-mini:

Python
TypeScript
JavaScript
curl

import Groq from "groq-sdk";

const groq = new Groq();

export async function main() {
    const completion = await groq.chat.completions.create({
      messages: [
        {
          role: "user",
          content: "What is the current weather in Tokyo?",
        },
      ],
      // Change model to compound-beta to use agentic tooling
      // model: "llama-3.3-70b-versatile",
      model: "compound-beta",
    });

    console.log(completion.choices[0]?.message?.content || "");
    // Print all tool calls
    // console.log(completion.choices[0]?.message?.executed_tools || "");
}

main();
And that's it!


When the API is called, it will intelligently decide when to use search or code execution to best answer the user's query. These tool calls are performed on the server side, so no additional setup is required on your part to use agentic tooling.


In the above example, the API will use its build in web search tool to find the current weather in Tokyo. If you didn't use agentic tooling, you might have needed to add your own custom tools to make API requests to a weather service, then perform multiple API calls to Groq to get a final result. Instead, with agentic tooling, you can get a final result with a single API call.

Use Cases
Compound-beta excels at a wide range of use cases, particularly when real-time information is required.

Real-time Fact Checker and News Agent
Your application needs to answer questions or provide information that requires up-to-the-minute knowledge, such as:

Latest news
Current stock prices
Recent events
Weather updates
Building and maintaining your own web scraping or search API integration is complex and time-consuming.

Solution with Compound Beta
Simply send the user's query to compound-beta. If the query requires current information beyond its training data, it will automatically trigger its built-in web search tool (powered by Tavily) to fetch relevant, live data before formulating the answer.

Why It's Great
Get access to real-time information instantly without writing any extra code for search integration
Leverage Groq's speed for a real-time, responsive experience
Code Example
Python
TypeScript
JavaScript

import Groq from "groq-sdk";

const groq = new Groq();

export async function main() {
  const user_query = "What were the main highlights from the latest Apple keynote event?"
  // Or: "What's the current weather in San Francisco?"
  // Or: "Summarize the latest developments in fusion energy research this week."

  const completion = await groq.chat.completions.create({
    messages: [
      {
        role: "user",
        content: user_query,
      },
    ],
    // The *only* change needed: Specify the compound model!
    model: "compound-beta",
  });

  console.log(`Query: ${user_query}`);
  console.log(`Compound Beta Response:\n${completion.choices[0]?.message?.content || ""}`);

  // You might also inspect chat_completion.choices[0].message.executed_tools
  // if you want to see if/which tool was used, though it's not necessary.
}

main();
Natural Language Calculator and Code Extractor
You want users to perform calculations, run simple data manipulations, or execute small code snippets using natural language commands within your application, without building a dedicated parser or execution environment.

Solution with Compound Beta
Frame the user's request as a task involving computation or code. compound-beta-mini can recognize these requests and use its secure code execution tool to compute the result.

Why It's Great
Effortlessly add computational capabilities
Users can ask things like:
"What's 15% of $540?"
"Calculate the standard deviation of [10, 12, 11, 15, 13]"
"Run this python code: print('Hello from Compound Beta!')"
Code Example
Python
TypeScript
JavaScript

import Groq from "groq-sdk";

const groq = new Groq();

export async function main() {
  // Example 1: Calculation
  const computationQuery = "Calculate the monthly payment for a $30,000 loan over 5 years at 6% annual interest.";

  // Example 2: Simple code execution
  const codeQuery = "What is the output of this Python code snippet: `data = {'a': 1, 'b': 2}; print(data.keys())`";

  // Choose one query to run
  const selectedQuery = computationQuery;

  const completion = await groq.chat.completions.create({
    messages: [
      {
        role: "system",
        content: "You are a helpful assistant capable of performing calculations and executing simple code when asked.",
      },
      {
        role: "user",
        content: selectedQuery,
      }
    ],
    // Use the compound model
    model: "compound-beta-mini",
  });

  console.log(`Query: ${selectedQuery}`);
  console.log(`Compound Beta Response:\n${completion.choices[0]?.message?.content || ""}`);
}

main();
Code Debugging Assistant
Developers often need quick help understanding error messages or testing small code fixes. Searching documentation or running snippets requires switching contexts.

Solution with Compound Beta
Users can paste an error message and ask for explanations or potential causes. Compound Beta Mini might use web search to find recent discussions or documentation about that specific error. Alternatively, users can provide a code snippet and ask "What's wrong with this code?" or "Will this Python code run: ...?". It can use code execution to test simple, self-contained snippets.

Why It's Great
Provides a unified interface for getting code help
Potentially draws on live web data for new errors
Executes code directly for validation
Speeds up the debugging process
Note: compound-beta-mini uses one tool per turn, so it might search OR execute, not both simultaneously in one response.

Code Example
Python
TypeScript
JavaScript

import Groq from "groq-sdk";

const groq = new Groq();

export async function main() {
  // Example 1: Error Explanation (might trigger search)
  const debugQuerySearch = "I'm getting a 'Kubernetes CrashLoopBackOff' error on my pod. What are the common causes based on recent discussions?";

  // Example 2: Code Check (might trigger code execution)
  const debugQueryExec = "Will this Python code raise an error? `import numpy as np; a = np.array([1,2]); b = np.array([3,4,5]); print(a+b)`";

  // Choose one query to run
  const selectedQuery = debugQueryExec;

  const completion = await groq.chat.completions.create({
    messages: [
      {
        role: "system",
        content: "You are a helpful coding assistant. You can explain errors, potentially searching for recent information, or check simple code snippets by executing them.",
      },
      {
        role: "user",
        content: selectedQuery,
      }
    ],
    // Use the compound model
    model: "compound-beta-mini",
  });

  console.log(`Query: ${selectedQuery}`);
  console.log(`Compound Beta Response:\n${completion.choices[0]?.message?.content || ""}`);
}

main();
