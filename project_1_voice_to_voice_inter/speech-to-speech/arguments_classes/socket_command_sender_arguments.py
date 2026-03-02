from dataclasses import dataclass, field


@dataclass
class SocketCommandSenderArguments:
    cmd_host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host for command sender socket. Default is '0.0.0.0'."},
    )
    cmd_port: int = field(
        default=12347,
        metadata={"help": "Port for command sender socket. Default is 12347."},
    )
