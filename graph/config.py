
class Config():

    def __init__(self, config=None):

        if config is not None:
            for (key, val) in config.items():
                setattr(self, key, val)
