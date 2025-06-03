// -----------------------
// -- example-cramit.js --
// --------------------------------------------------------------------------------
// this is an example of how to use the cramit function
// first we import the cramit function and LocalEmbeddingModel class
// then we initialize the model once
// then we setup the documents array with a text
// then we call the cramit function with the documents, model, and options object
// the options object is optional
//
// the cramit function is faster than the chunkit function, but it is less accurate
// useful for quickly chunking text, but not for exact semantic chunking
// --------------------------------------------------------------------------------

import { LocalEmbeddingModel, cramit } from "../chunkit.js"; // this is typically just "import { LocalEmbeddingModel, cramit } from 'semantic-chunking';", but this is a local test
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
const model = new LocalEmbeddingModel();
await model.initialize(
  "nomic-ai/nomic-embed-text-v1.5", // model name
  "q8", // dtype
  "../models", // localModelPath
  "../models" // modelCacheDir
);

let myTestChunks = await cramit(
  documents,
  model, // Pass the initialized model
  {
    logging: false,
    maxTokenSize: 300,
    returnEmbedding: false,
    returnTokenLength: true,
  }
);

// end timeing
const endTime = performance.now();

// calculate tracked time in seconds
let trackedTimeSeconds = (endTime - startTime) / 1000;
trackedTimeSeconds = parseFloat(trackedTimeSeconds.toFixed(2));

console.log("\n\n\n");
console.log("myTestChunks:");
console.log(myTestChunks);
console.log("length: " + myTestChunks.length);
console.log("trackedTimeSeconds: " + trackedTimeSeconds);
