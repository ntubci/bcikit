from types import SimpleNamespace


class BcikitConfig(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, BcikitConfig(value))
            else:
                self.__setattr__(key, value)
