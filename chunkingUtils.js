import { cosineSimilarity } from "./similarityUtils.js";

// -----------------------------------------------------------
// -- Function to create chunks of text based on similarity --
// -----------------------------------------------------------
export async function createChunks(
  sentences,
  similarities,
  maxTokenSize,
  similarityThreshold,
  logging,
  model
) {
  let chunks = [];
  let currentChunk = [sentences[0]];

  if (logging) {
    console.log("Initial sentence:", sentences[0]);
  }

  for (let i = 1; i < sentences.length; i++) {
    const nextSentence = sentences[i];

    // For cramit (when similarities is null), only check token size
    if (!similarities) {
      const currentChunkText = currentChunk.join(" ");
      const currentTokenized = await model.tokenize(currentChunkText);
      const currentChunkSize = currentTokenized.size;
      const nextSentenceTokenized = await model.tokenize(nextSentence);
      const nextSentenceTokenCount = nextSentenceTokenized.size;

      if (currentChunkSize + nextSentenceTokenCount <= maxTokenSize) {
        currentChunk.push(nextSentence);
      } else {
        chunks.push(currentChunkText);
        currentChunk = [nextSentence];
      }
      continue;
    }

    // Check similarity first for chunkit
    if (similarities[i - 1] >= similarityThreshold) {
      if (logging) {
        console.log(
          `Adding sentence ${i} with similarity ${similarities[i - 1]}`
        );
      }

      // Then check token size
      const currentChunkText = currentChunk.join(" ");
      const currentTokenized = await model.tokenize(currentChunkText);
      const currentChunkSize = currentTokenized.size;
      const nextSentenceTokenized = await model.tokenize(nextSentence);
      const nextSentenceTokenCount = nextSentenceTokenized.size;

      if (currentChunkSize + nextSentenceTokenCount <= maxTokenSize) {
        currentChunk.push(nextSentence);
      } else {
        chunks.push(currentChunkText);
        currentChunk = [nextSentence];
      }
    } else {
      if (logging) {
        console.log(
          `Starting new chunk at sentence ${i}, similarity was ${
            similarities[i - 1]
          }`
        );
      }
      chunks.push(currentChunk.join(" "));
      currentChunk = [nextSentence];
    }
  }

  if (currentChunk.length > 0) {
    chunks.push(currentChunk.join(" "));
  }

  return chunks;
}

// --------------------------------------------------------------
// -- Optimize and Rebalance Chunks (optionally use Similarity) --
// --------------------------------------------------------------
export async function optimizeAndRebalanceChunks(
  combinedChunks,
  model,
  maxTokenSize,
  combineChunksSimilarityThreshold = 0.5
) {
  let optimizedChunks = [];
  let currentChunkText = "";
  let currentChunkTokenCount = 0;
  let currentEmbedding = null;

  for (let index = 0; index < combinedChunks.length; index++) {
    const chunk = combinedChunks[index];
    const chunkTokenized = await model.tokenize(chunk);
    const chunkTokenCount = chunkTokenized.size;

    if (
      currentChunkText &&
      currentChunkTokenCount + chunkTokenCount <= maxTokenSize
    ) {
      const nextEmbedding = await model.createEmbedding(chunk);
      const similarity = currentEmbedding
        ? cosineSimilarity(currentEmbedding, nextEmbedding)
        : 0;

      if (similarity >= combineChunksSimilarityThreshold) {
        currentChunkText += " " + chunk;
        currentChunkTokenCount += chunkTokenCount;
        currentEmbedding = nextEmbedding;
        continue;
      }
    }

    if (currentChunkText) optimizedChunks.push(currentChunkText);
    currentChunkText = chunk;
    currentChunkTokenCount = chunkTokenCount;
    currentEmbedding = await model.createEmbedding(chunk);
  }

  if (currentChunkText) optimizedChunks.push(currentChunkText);

  return optimizedChunks.filter((chunk) => chunk);
}

// ------------------------------------------------
// -- Helper function to apply prefix to a chunk --
// ------------------------------------------------
export function applyPrefixToChunk(chunkPrefix, chunk) {
  if (chunkPrefix && chunkPrefix.trim()) {
    return `${chunkPrefix}: ${chunk}`;
  }
  return chunk;
}
