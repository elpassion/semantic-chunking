// ------------------------
// -- example-openai-chunkit.js --
// -------------------------------------------------------------------------------
// this is an example of how to use the chunkit function with OpenAI embeddings
// first we import the chunkit function and OpenAIEmbedding class
// then we initialize the OpenAI client and model once
// then we setup the documents array with text files
// then we call the chunkit function with the documents array, model, and options object
// the options object is optional, use it to customize the chunking process
//
// NOTE: You need to set your OPENAI_API_KEY environment variable
// You also need to install the openai package: npm install openai
// -------------------------------------------------------------------------------

import { OpenAIEmbedding, chunkit } from "../chunkit.js"; // this is typically just "import { OpenAIEmbedding, chunkit } from 'semantic-chunking';", but this is a local test
import OpenAI from "openai";
import fs from "fs";
import { fileURLToPath } from "url";
import { dirname, resolve } from "path";

// Get current file's directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Check for OpenAI API key
if (!process.env.OPENAI_API_KEY) {
  console.error("Please set your OPENAI_API_KEY environment variable");
  process.exit(1);
}

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// initialize documents array
let documents = [];
let textFiles = ["example.txt", "different.txt", "similar.txt"].map((file) =>
  resolve(__dirname, file)
);

// read each text file and add it to the documents array
for (const textFile of textFiles) {
  documents.push({
    document_name: textFile,
    document_text: await fs.promises.readFile(textFile, "utf8"),
  });
}

// start timing
const startTime = performance.now();

// Initialize the OpenAI embedding model
const model = new OpenAIEmbedding(openai);
await model.initialize("text-embedding-3-small"); // or "text-embedding-3-large" for higher quality

let myTestChunks = await chunkit(
  documents,
  model, // Pass the initialized OpenAI model
  {
    logging: false,
    maxTokenSize: 300,
    similarityThreshold: 0.5,
    dynamicThresholdLowerBound: 0.4,
    dynamicThresholdUpperBound: 0.8,
    numSimilaritySentencesLookahead: 3,
    combineChunks: true, // enable rebalancing
    combineChunksSimilarityThreshold: 0.7,
    returnTokenLength: true,
    returnEmbedding: false,
  }
);

// end timing
const endTime = performance.now();

// calculate tracked time in seconds
let trackedTimeSeconds = (endTime - startTime) / 1000;
trackedTimeSeconds = parseFloat(trackedTimeSeconds.toFixed(2));

console.log("\n\n");
console.log("myTestChunks:");
console.log(myTestChunks);
console.log("length: " + myTestChunks.length);
console.log("trackedTimeSeconds: " + trackedTimeSeconds);
console.log("Model info:", model.getModelInfo());
