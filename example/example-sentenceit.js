// -----------------------
// -- example-sentenceit.js --
// --------------------------------------------------------------------------------
// this is an example of how to use the sentenceit function
// first we import the sentenceit function and EmbeddingModel class
// then we initialize the model once
// then we setup the documents array with a text
// then we call the sentenceit function with the documents, model, and options object
// the options object is optional
//
// the sentenceit function splits text into sentences and optionally returns embeddings
// useful for splitting text into individual sentences with semantic understanding
// --------------------------------------------------------------------------------

import { EmbeddingModel, sentenceit } from "../chunkit.js"; // this is typically just "import { EmbeddingModel, sentenceit } from 'semantic-chunking';", but this is a local test
import fs from "fs";
import { fileURLToPath } from "url";
import { dirname, resolve } from "path";

// Get current file's directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// initialize documents array
let documents = [];
let textFiles = ["example3.txt"].map((file) => resolve(__dirname, file));

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
  "fp32", // dtype
  "../models", // localModelPath
  "../models" // modelCacheDir
);

let myTestSentences = await sentenceit(
  documents,
  model, // Pass the initialized model
  {
    logging: false,
    returnEmbedding: true,
  }
);

// end timeing
const endTime = performance.now();

// calculate tracked time in seconds
let trackedTimeSeconds = (endTime - startTime) / 1000;
trackedTimeSeconds = parseFloat(trackedTimeSeconds.toFixed(2));

console.log("\n\n\n");
console.log("myTestSentences:");
console.log(myTestSentences);
console.log("length: " + myTestSentences.length);
console.log("trackedTimeSeconds: " + trackedTimeSeconds);
