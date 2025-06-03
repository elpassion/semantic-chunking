import { env, pipeline, AutoTokenizer } from "@huggingface/transformers";
import { LRUCache } from "lru-cache";

// --------------------------
// -- LocalEmbeddingModel class --
// --------------------------
export class LocalEmbeddingModel {
  constructor() {
    this.tokenizer = null;
    this.generateEmbedding = null;
    this.modelName = null;
    this.dtype = null;
    this.embeddingCache = new LRUCache({
      max: 500,
      maxSize: 50_000_000,
      sizeCalculation: (value, key) => {
        return value.length * 4 + key.length;
      },
      ttl: 1000 * 60 * 60,
    });
  }

  async initialize(
    onnxEmbeddingModel,
    dtype = "fp32",
    localModelPath = null,
    modelCacheDir = null
  ) {
    // Configure environment
    env.allowRemoteModels = true;
    if (localModelPath) env.localModelPath = localModelPath;
    if (modelCacheDir) env.cacheDir = modelCacheDir;

    this.tokenizer = await AutoTokenizer.from_pretrained(onnxEmbeddingModel);
    this.generateEmbedding = await pipeline(
      "feature-extraction",
      onnxEmbeddingModel,
      {
        dtype: dtype,
      }
    );

    this.modelName = onnxEmbeddingModel;
    this.dtype = dtype;
    this.embeddingCache.clear();

    return {
      modelName: onnxEmbeddingModel,
      dtype: dtype,
    };
  }

  async createEmbedding(text) {
    if (!this.generateEmbedding) {
      throw new Error("Model not initialized. Call initialize() first.");
    }

    const cached = this.embeddingCache.get(text);
    if (cached) {
      return cached;
    }

    const embeddings = await this.generateEmbedding(text, {
      pooling: "mean",
      normalize: true,
    });
    this.embeddingCache.set(text, embeddings.data);
    return embeddings.data;
  }

  async tokenize(text, options = {}) {
    if (!this.tokenizer) {
      throw new Error("Model not initialized. Call initialize() first.");
    }
    const tokenized = await this.tokenizer(text, options);
    return {
      size: tokenized.input_ids.size,
    };
  }

  getModelInfo() {
    return {
      modelName: this.modelName,
      dtype: this.dtype,
    };
  }
}

// --------------------------
// -- OpenAIEmbedding class --
// --------------------------
export class OpenAIEmbedding {
  constructor(openaiClient) {
    if (!openaiClient) {
      throw new Error("OpenAI client is required in constructor");
    }
    this.openaiClient = openaiClient;
    this.modelName = null;
    this.embeddingCache = new LRUCache({
      max: 500,
      maxSize: 50_000_000,
      sizeCalculation: (value, key) => {
        return value.length * 4 + key.length;
      },
      ttl: 1000 * 60 * 60,
    });
  }

  async initialize(modelName = "text-embedding-3-small") {
    this.modelName = modelName;
    this.embeddingCache.clear();

    return {
      modelName: modelName,
      dtype: "api", // API-based, no dtype
    };
  }

  async createEmbedding(text) {
    if (!this.modelName) {
      throw new Error("Model not initialized. Call initialize() first.");
    }

    const cached = this.embeddingCache.get(text);
    if (cached) {
      return cached;
    }

    try {
      const response = await this.openaiClient.embeddings.create({
        model: this.modelName,
        input: text,
      });

      const embedding = response.data[0].embedding;
      this.embeddingCache.set(text, embedding);
      return embedding;
    } catch (error) {
      throw new Error(`OpenAI API error: ${error.message}`);
    }
  }

  async tokenize(text, options = {}) {
    if (!this.modelName) {
      throw new Error("Model not initialized. Call initialize() first.");
    }

    // Rough approximation for tokenization since OpenAI doesn't provide a direct tokenization endpoint
    // This is a simplified estimation based on common tokenization patterns
    // For more accurate results, consider using a library like 'tiktoken' for OpenAI tokenization
    const approximateTokenCount = Math.ceil(text.length / 4); // Rough estimation: 1 token ≈ 4 characters

    return {
      size: approximateTokenCount,
    };
  }

  getModelInfo() {
    return {
      modelName: this.modelName,
      dtype: "api",
    };
  }
}
