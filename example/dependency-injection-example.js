// ------------------------
// -- dependency-injection-example.js --
// -------------------------------------------------------------------------------
// This example demonstrates dependency injection with different embedding models
// Shows how LocalEmbeddingModel and OpenAIEmbedding can be used interchangeably
// The processDocuments function accepts any model that implements the common interface
// -------------------------------------------------------------------------------

import { LocalEmbeddingModel, OpenAIEmbedding, chunkit } from "../chunkit.js";
import OpenAI from "openai";

// Common processing function that works with any embedding model
async function processDocuments(model, documents, options = {}) {
  console.log(`Processing with model: ${model.getModelInfo().modelName}`);

  const chunks = await chunkit(documents, model, {
    maxTokenSize: 300,
    similarityThreshold: 0.5,
    ...options,
  });

  console.log(`Created ${chunks.length} chunks`);
  return chunks;
}

// Sample documents
const documents = [
  {
    document_name: "AI Overview",
    document_text:
      "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. Deep learning, a subset of machine learning, uses neural networks with multiple layers to model and understand complex patterns in data.",
  },
  {
    document_name: "Technology Trends",
    document_text:
      "Cloud computing has revolutionized how businesses store and process data. Edge computing brings computation closer to data sources, reducing latency. Quantum computing promises to solve complex problems that are intractable for classical computers. Blockchain technology provides decentralized and secure transaction recording.",
  },
];

async function main() {
  console.log("=== Dependency Injection Example ===\n");

  // Example 1: Using LocalEmbeddingModel
  console.log("1. Using Local Embedding Model:");
  const localModel = new LocalEmbeddingModel();
  await localModel.initialize("Xenova/all-MiniLM-L6-v2", "q8");

  const localChunks = await processDocuments(localModel, documents);
  console.log("Local model info:", localModel.getModelInfo());
  console.log();

  // Example 2: Using OpenAI Embeddings (if API key is available)
  if (process.env.OPENAI_API_KEY) {
    console.log("2. Using OpenAI Embedding Model:");
    const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    const openaiModel = new OpenAIEmbedding(openai);
    await openaiModel.initialize("text-embedding-3-small");

    const openaiChunks = await processDocuments(openaiModel, documents);
    console.log("OpenAI model info:", openaiModel.getModelInfo());
    console.log();
  } else {
    console.log("2. Skipping OpenAI example (OPENAI_API_KEY not set)");
    console.log();
  }

  // Example 3: Model factory pattern
  console.log("3. Using Model Factory Pattern:");
  const models = [];

  // Always add local model
  models.push({
    name: "Local Model",
    instance: await createLocalModel(),
  });

  // Add OpenAI model if available
  if (process.env.OPENAI_API_KEY) {
    models.push({
      name: "OpenAI Model",
      instance: await createOpenAIModel(),
    });
  }

  // Process with all available models
  for (const modelConfig of models) {
    console.log(`Processing with ${modelConfig.name}:`);
    await processDocuments(modelConfig.instance, documents.slice(0, 1)); // Process just one document for demo
  }
}

// Factory functions
async function createLocalModel() {
  const model = new LocalEmbeddingModel();
  await model.initialize("Xenova/all-MiniLM-L6-v2");
  return model;
}

async function createOpenAIModel() {
  const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const model = new OpenAIEmbedding(openai);
  await model.initialize("text-embedding-3-small");
  return model;
}

// Run the example
main().catch(console.error);
