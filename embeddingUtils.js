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
