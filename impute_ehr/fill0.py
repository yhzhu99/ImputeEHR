class Fill0Impute:
    def __init__(self, ds):
        self.ds = ds

    def execute(self):
        return self.ds.fillna(0)
