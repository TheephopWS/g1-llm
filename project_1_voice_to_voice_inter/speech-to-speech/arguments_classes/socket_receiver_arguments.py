from dataclasses import dataclass, field


@dataclass
class SocketReceiverArguments:
    recv_host: str = field(
        default="localhost",
        metadata={"help": "Host IP for the socket receiver. Default is 'localhost'."},
    )
    recv_port: int = field(
        default=12345,
        metadata={"help": "Port for the socket receiver. Default is 12345."},
    )
    chunk_size: int = field(
        default=1024,
        metadata={"help": "Size of each audio data chunk in bytes. Default is 1024."},
    )
