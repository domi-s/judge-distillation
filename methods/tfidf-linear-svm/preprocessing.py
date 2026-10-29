import re


def normalise_code(code: str) -> str:
  code = re.sub(r"//.*", "", code)
  code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
  return re.sub(r"\s+", " ", code).strip()


def tokenise_code(code: str) -> list[str]:
  code = normalise_code(code)
  pattern = r"[A-Za-z_]\w*|\d+|[+\-*/%=<>!&|^~]+|[;,\{\}\[\]\(\)]"
  return re.findall(pattern, code)
