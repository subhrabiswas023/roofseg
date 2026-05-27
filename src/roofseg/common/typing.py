from dataclasses import asdict, dataclass, fields

type JsonValue = (
    dict[str, JsonValue] | list[JsonValue] | int | float | str | bool | None
)
type JsonDict = dict[str, JsonValue]
type JsonArray = list[JsonValue]
type JsonData = JsonDict | JsonArray

type StateDict = dict[str, object]

@dataclass(frozen=True)
class Dataclass:
    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict[T: Dataclass](cls: type[T], data: JsonDict) -> T:
        field_types = {f.name: f.type for f in fields(cls)}

        init_kwargs = {}
        for key, value in data.items():
            if key not in field_types:
                continue

            expected_type = field_types[key]
            if (
                isinstance(expected_type, type)
                and issubclass(expected_type, Dataclass)
                and isinstance(value, dict)
            ):
                init_kwargs[key] = expected_type.from_dict(value)
            else:
                init_kwargs[key] = value

        return cls(**init_kwargs)
