// ------------------------
// -- example-chunkit.js --
// -------------------------------------------------------------------------------
// this is an example of how to use the chunkit function
// first we import the chunkit function and EmbeddingModel class
// then we initialize the model once
// then we setup the documents array with text files
// then we call the chunkit function with the documents array, model, and options object
// the options object is optional, use it to customize the chunking process
// -------------------------------------------------------------------------------

import { EmbeddingModel, chunkit } from "../chunkit.js"; // this is typically just "import { EmbeddingModel, chunkit } from 'semantic-chunking';", but this is a local test
import fs from "fs";
import { fileURLToPath } from "url";
import { dirname, resolve } from "path";

// Get current file's directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

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

// Initialize the model once
const model = new EmbeddingModel();
await model.initialize(
  "Xenova/all-MiniLM-L6-v2", // model name
  "q8", // dtype
  "../models", // localModelPath
  "../models" // modelCacheDir
);

let myTestChunks = await chunkit(
  documents,
  model, // Pass the initialized model
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

// end timeing
const endTime = performance.now();

// calculate tracked time in seconds
let trackedTimeSeconds = (endTime - startTime) / 1000;
trackedTimeSeconds = parseFloat(trackedTimeSeconds.toFixed(2));

console.log("\n\n");
console.log("myTestChunks:");
console.log(myTestChunks);
console.log("length: " + myTestChunks.length);
console.log("trackedTimeSeconds: " + trackedTimeSeconds);
