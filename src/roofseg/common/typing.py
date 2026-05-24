from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Dataclass:
    def asdict(self) -> dict[str, object]:
        return asdict(self)
