class DummyReader:
    def generate(self, prompt: str) -> str:
        _ = prompt
        return ""


class EchoReader:
    def generate(self, prompt: str) -> str:
        return prompt


def build_reader(reader_type: str):
    if reader_type == "dummy":
        return DummyReader()
    if reader_type == "echo":
        return EchoReader()
    raise ValueError(f"Unsupported reader type: {reader_type}")
