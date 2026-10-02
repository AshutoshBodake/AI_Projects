using Microsoft.Data.SqlClient;
using Qdrant.Client;
using Qdrant.Client.Grpc;

class Program
{
    static async Task Main(string[] args)
    {
        // 1. Initialize Qdrant Client ( connects to local Docker instance )
        var qdrantClient = new QdrantClient("localhost");
        string collectionName = "prompts";

        // Create collection if it doesn't exist (e.g., 4-dimensional vector size)
        var collections = await qdrantClient.ListCollectionsAsync();
        if (!collections.Contains(collectionName))
        {
            await qdrantClient.CreateCollectionAsync(
                collectionName,
                new VectorParams { Size = 4, Distance = Distance.Cosine }
            );
        }

        // 2. Read records from SQL Database
        string sqlConnString = "Server=localhost;Database=PromptyMarketplace;Integrated Security=true;TrustServerCertificate=true;";
        var points = new List<PointStruct>();

        using (var connection = new SqlConnection(sqlConnString))
        {
            await connection.OpenAsync();
            string query = "SELECT PromptId, Title, Description FROM Prompts"; // Pre-calculated or mock 4D vectors stored in SQL

            using (var command = new SqlCommand(query, connection))
            using (var reader = await command.ExecuteReaderAsync())
            {
                while (await reader.ReadAsync())
                {
                    ulong id = (ulong)reader.GetInt32(0);
                    string title = reader.GetString(1);
                    string description = reader.GetString(2);

                    // Fixed: Replaced your previous reader.GetFloat call. 
                    // Because your SQL table returns strings here, we use a placeholder 4D vector.
                    // (In production, you will generate these 4 values using an LLM / embedding model)
                    float[] vector = [0.1f, 0.5f, 0.75f, 1.0f];

                    // Map to Qdrant PointStruct with SQL metadata payload
                    points.Add(new PointStruct
                    {
                        Id = id,
                        Vectors = vector,
                        Payload = { ["Title"] = title, ["Description"] = description }
                    });
                }
            }
        }

        // 3. Upsert records into Qdrant
        if (points.Count > 0)
        {
            await qdrantClient.UpsertAsync(collectionName, points);
            Console.WriteLine($"Successfully synced {points.Count} rows from SQL to Qdrant.");
        }

        // 4. Perform a Vector Similarity Search in Qdrant
        float[] queryVector = [0.1f, 0.2f, 0.3f, 0.4f];
        var searchResults = await qdrantClient.SearchAsync(
            collectionName: collectionName,
            vector: queryVector,
            limit: 3
        );

        Console.WriteLine("\nSearch Results:");
        foreach (var hit in searchResults)
        {
            // Fixed: Changed from "Name" to "Title" to match your payload structure
            Console.WriteLine($"Found: {hit.Payload["Title"]} with Score: {hit.Score}");
            Console.WriteLine($"Description: {hit.Payload["Description"]}\n");
        }


        // ================= NEW LINES ADDED BELOW =================

        Console.WriteLine("--- Interactive Vector Search Mode ---");
        while (true)
        {
            Console.Write("Enter your search prompt (or type 'exit' to quit): ");
            string? userInput = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(userInput) || userInput.Trim().Equals("exit", StringComparison.OrdinalIgnoreCase))
            {
                break;
            }

            // In production, you would pass the 'userInput' string to an embedding model 
            // (like OpenAI, Ollama, or HuggingFace) to generate a real 4D numerical array.
            // For now, we generate a mock vector based on the string length to vary the results.
            float mockFactor = (float)userInput.Length / 100f;
            float[] interactiveQueryVector = [mockFactor, 0.2f, 0.5f, 0.8f];

            Console.WriteLine($"\nSearching Qdrant for: \"{userInput}\"...");

            var interactiveResults = await qdrantClient.SearchAsync(
                collectionName: collectionName,
                vector: interactiveQueryVector,
                limit: 3
            );

            if (interactiveResults.Count == 0)
            {
                Console.WriteLine("No matching results found.\n");
                continue;
            }

            foreach (var hit in interactiveResults)
            {
                Console.WriteLine($"Found: {hit.Payload["Title"]} | Match Score: {hit.Score}");
                Console.WriteLine($"Description: {hit.Payload["Description"]}\n");
            }
        }

        Console.WriteLine("Program exited.");
    }
}
