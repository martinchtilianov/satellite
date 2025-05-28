var satellite = new Satellite.Satellite();
await satellite.Connect();

Console.WriteLine("Press Ctrl+C to quit...");
await Task.Delay(Timeout.Infinite);