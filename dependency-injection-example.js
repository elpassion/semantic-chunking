import { LocalEmbeddingModel, chunkit } from "./chunkit.js";

// Example demonstrating dependency injection with semantic chunking

async function main() {
  try {
    // Create and initialize the model outside of the chunking function
    const model = new LocalEmbeddingModel();
    await model.initialize(
      "Xenova/all-MiniLM-L6-v2", // Model name
      "q8", // Data type
      "./models", // Local model path
      "./models" // Model cache directory
    );

    console.log("Model initialized:", model.getModelInfo());

    // Sample documents to chunk
    const documents = [
      {
        document_name: "AI and Machine Learning",
        document_text: `
          Artificial Intelligence (AI) is a rapidly evolving field that encompasses machine learning, 
          deep learning, and neural networks. Machine learning algorithms enable computers to learn 
          from data without being explicitly programmed. Deep learning, a subset of machine learning, 
          uses neural networks with multiple layers to model and understand complex patterns in data.
          
          Natural Language Processing (NLP) is another important branch of AI that focuses on the 
          interaction between computers and human language. NLP techniques are used in applications 
          like chatbots, language translation, and sentiment analysis.
          
          Computer vision is yet another AI domain that enables machines to interpret and understand 
          visual information from the world. This technology is used in autonomous vehicles, medical 
          imaging, and facial recognition systems.
        `,
      },
    ];

    // Use the new API with the pre-initialized model
    const chunks = await chunkit(
      documents,
      model, // Pass the initialized model as second parameter
      {
        logging: true,
        maxTokenSize: 100,
        similarityThreshold: 0.5,
        returnEmbedding: true,
        returnTokenLength: true,
      }
    );

    console.log("\n=== CHUNKING RESULTS ===");
    chunks.forEach((chunk, index) => {
      console.log(`\nChunk ${index + 1}:`);
      console.log(`Model: ${chunk.model_name} (${chunk.dtype})`);
      console.log(`Token Length: ${chunk.token_length}`);
      console.log(`Text: ${chunk.text.substring(0, 100)}...`);
      console.log(`Embedding dimensions: ${chunk.embedding.length}`);
    });

    // Example of reusing the same model with different documents or configurations
    console.log("\n=== REUSING MODEL WITH DIFFERENT CONFIG ===");

    const moreDocuments = [
      {
        document_name: "Technology Evolution",
        document_text: `
          The evolution of technology has accelerated dramatically in recent decades. From the invention 
          of the personal computer to the rise of the internet, each technological breakthrough has 
          fundamentally changed how we work, communicate, and live.
          
          Mobile devices have revolutionized personal computing, putting powerful computers in everyone's 
          pockets. Cloud computing has transformed how businesses operate, allowing for scalable and 
          flexible IT infrastructure.
        `,
      },
    ];

    const moreChunks = await chunkit(
      moreDocuments,
      model, // Reuse the same model instance
      {
        maxTokenSize: 150,
        similarityThreshold: 0.3,
        returnEmbedding: false, // Skip embeddings this time
        returnTokenLength: true,
      }
    );

    console.log("\nSecond chunking results:");
    moreChunks.forEach((chunk, index) => {
      console.log(`\nChunk ${index + 1}: ${chunk.text.substring(0, 80)}...`);
    });
  } catch (error) {
    console.error("Error:", error);
  }
}

main();
