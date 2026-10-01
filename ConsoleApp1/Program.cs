using Microsoft.Data.SqlClient;
using Qdrant.Client;
using Qdrant.Client.Grpc;

class Program
{
    static async Task Main(string[] args)
    {
        // 1. Initialize Qdrant Client ( connects to local Docker instance )
        var qdrantClient = new QdrantClient("localhost");
        string collectionName = "products";

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
        string sqlConnString = "Server=localhost;Database=TestDb;Integrated Security=true;TrustServerCertificate=true;";
        var points = new List<PointStruct>();

        using (var connection = new SqlConnection(sqlConnString))
        {
            await connection.OpenAsync();
            string query = "SELECT Id, Name, Vec1, Vec2, Vec3, Vec4 FROM Products"; // Pre-calculated or mock 4D vectors stored in SQL

            using (var command = new SqlCommand(query, connection))
            using (var reader = await command.ExecuteReaderAsync())
            {
                while (await reader.ReadAsync())
                {
                    ulong id = (ulong)reader.GetInt64(0);
                    string name = reader.GetString(1);

                    // Read vector elements from SQL columns
                    float[] vector = [
                        reader.GetFloat(2),
                        reader.GetFloat(3),
                        reader.GetFloat(4),
                        reader.GetFloat(5)
                    ];

                    // Map to Qdrant PointStruct with SQL metadata payload
                    points.Add(new PointStruct
                    {
                        Id = id,
                        Vectors = vector,
                        Payload = { ["Name"] = name }
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
            Console.WriteLine($"Found: {hit.Payload["Name"]} with Score: {hit.Score}");
        }
    }
}
