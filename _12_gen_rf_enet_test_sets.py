
from _11_test import TestPredictor

class RModelDataMaker(TestPredictor):
    def __init__(self, batchstring):
        super().__init__(batchstring, task = 'multi')

    def run(self):
        self.compile_structured_data()

if __name__ == "__main__":
    self = RModelDataMaker()
