from typing import List

PAD = "<pad>"
UNK = "<unk>"

class Vocab:
    def __init__(self, specials: List[str] = None, add_base_specials: bool = True):
        """
        - add_base_specials=True：用于词表（会自动加入 <pad>/<unk>）
        - add_base_specials=False：用于标签表（不自动加入 <pad>/<unk>）
        """
        self.itos = []
        self.stoi = {}
        base = [PAD, UNK] if add_base_specials else []
        extras = specials or []
        seen = set()
        for tok in base + extras:
            if tok not in seen:
                seen.add(tok)
                self.add(tok)

    def add(self, tok: str):
        if tok not in self.stoi:
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)

    def __len__(self):
        return len(self.itos)

    def id(self, tok: str) -> int:
        # 词表需要兜底；标签表我们会严格受控，不会出现未知标签
        return self.stoi.get(tok, self.stoi.get(UNK, 0))

    def token(self, idx: int) -> str:
        return self.itos[idx]
