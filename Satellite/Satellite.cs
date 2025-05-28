using System;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Satellite;
using MQTTnet;

namespace Satellite;

/*
 * The ESP boards have already been flashed with the necessary firmware for handling UART input and
   communicating with the server. Each ESP publishes and subscribes to two different topics that follow the scheme: 
   /pynqbridge/{moduleNumber}/send and /pynqbridge/{moduleNumber}/recv,
   respectively. As the ESP subscribes to a certain topic, whenever there is a new message, 
   it automatically receives it and sends it via UART to the PYNQ board. The PYNQ implements a FIFO
   (first-in-first-out) buffer in the UART channel from which the messages can be extracted. Keep in
   mind that the buffer is not infinite. Your task is to read and handle arriving messages on the UART
   channel of the PYNQ board.
   In order to send a message from PYNQ to server, follow the inverse path. First, send it to the ESP
   board via UART. It will then be forwarded to the server, given that the message follows the proper
   format.
 */


public class Satellite
{
    MqttClientFactory clientFactory;
    
    IMqttClient mqttClient79;
    IMqttClient mqttClient80;
    
    MqttClientOptions clientOptions79;
    MqttClientOptions clientOptions80;

    public Satellite()
    {
        clientFactory = new MqttClientFactory();
        
        clientOptions79 = new MqttClientOptionsBuilder()
                              .WithTcpServer(Credentials.Hostname)
                              .WithCredentials(Credentials.Module79User, Credentials.Module79Password)
                              .Build();
        
        clientOptions80 = new MqttClientOptionsBuilder()
            .WithTcpServer(Credentials.Hostname)
            .WithCredentials(Credentials.Module80User, Credentials.Module80Password)
            .Build();
        
        mqttClient79 = clientFactory.CreateMqttClient();
        mqttClient80 = clientFactory.CreateMqttClient();
        
        mqttClient79.ApplicationMessageReceivedAsync+= async e =>
        {
            Console.WriteLine($"[79->80] Message received: {Encoding.UTF8.GetString(e.ApplicationMessage.Payload)}");
            var applicationMessage = new MqttApplicationMessageBuilder()
                .WithTopic(Credentials.Module80RecvTopic)
                .WithPayload(e.ApplicationMessage.Payload)
                .Build();

            if (mqttClient80.IsConnected)
            {
                await mqttClient80.PublishAsync(applicationMessage, CancellationToken.None);
                Console.WriteLine("[79->80] Message forwarded.");
            }
            else
            {
                Console.WriteLine("[79->80] ERROR: mqttClient80 is not connected!");
            }
        };
        
        mqttClient80.ApplicationMessageReceivedAsync += async e =>
        {
            Console.WriteLine($"[80->79] Message received: {Encoding.UTF8.GetString(e.ApplicationMessage.Payload)}");

            var applicationMessage = new MqttApplicationMessageBuilder()
                .WithTopic(Credentials.Module79RecvTopic)
                .WithPayload(e.ApplicationMessage.Payload)
                .Build();

            if (mqttClient79.IsConnected)
            {
                await mqttClient79.PublishAsync(applicationMessage, CancellationToken.None);
                Console.WriteLine("[80->79] Message forwarded.");
            }
            else
            {
                Console.WriteLine("[80->79] ERROR: mqttClient79 is not connected!");
            }
        };
    }

    public async Task Connect()
    {
        Console.WriteLine("Connecting mqttClient79...");
        var response79 = await mqttClient79.ConnectAsync(clientOptions79);
        Console.WriteLine($"mqttClient79: {response79.ResponseInformation}");

        Console.WriteLine("Connecting mqttClient80...");
        var response80 = await mqttClient80.ConnectAsync(clientOptions80);
        Console.WriteLine($"mqttClient80: {response80.ResponseInformation}");

        Console.WriteLine("Subscribing to topics...");
        await mqttClient79.SubscribeAsync(Credentials.Module79SendTopic);
        await mqttClient80.SubscribeAsync(Credentials.Module80SendTopic);

        Console.WriteLine("Satellite system is ready! 🚀");
    }
}
