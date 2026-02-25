from dataclasses import dataclass, field


@dataclass
class SocketSenderArguments:
    send_host: str = field(
        default="localhost",
        metadata={"help": "Host IP for the socket sender. Default is 'localhost'."},
    )
    send_port: int = field(
        default=12346,
        metadata={"help": "Port for the socket sender. Default is 12346."},
    )
