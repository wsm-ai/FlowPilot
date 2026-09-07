import re


SERVER_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def is_valid_server_id(value: str) -> bool:
    return SERVER_ID_PATTERN.fullmatch(value) is not None
