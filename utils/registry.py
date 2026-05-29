from typing import Callable, TypeVar, Generic, Dict, Optional

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self) -> None:
        self._objects: Dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def wrapper(obj: T) -> T:
            if name in self._objects:
                raise ValueError(f"Object already registered: {name}")
            self._objects[name] = obj
            return obj
        return wrapper

    def get(self, name: str) -> Optional[T]:
        return self._objects.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._objects

    def keys(self):
        return self._objects.keys()