namespace Satellite;

public static class Credentials
{
    public static string Module79User { get; } = "robot_79_1";
    public static string Module79Password { get; } = "faytWakUm0";
    public static string Module79RecvTopic { get; } = "/pynqbridge/79/recv";
    public static string Module79SendTopic { get; } = "/pynqbridge/79/send";
    
    public static string Module80User { get; } = "robot_80_1";
    public static string Module80Password { get; } = "afAjOtVa";
    public static string Hostname { get; } = "mqtt.ics.ele.tue.nl";
    public static string Module80RecvTopic { get; } = "/pynqbridge/80/recv";
    public static string Module80SendTopic { get; } = "/pynqbridge/80/send";
}