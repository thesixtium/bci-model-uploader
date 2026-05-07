import unittest

from src.classifierSource import ClassifierSource

class ClassifierSourceTest(unittest.TestCase):

    def test_update_differentClassifier(self):
        cs = ClassifierSource()

        cs.model = 1
        cs.newModel = 2

        cs.update()

        self.assertTrue( cs.isThereNewData() )

    def test_update_same(self):
        cs = ClassifierSource()

        cs.newModel = 1
        cs.model = 1

        cs.update()

        self.assertFalse( cs.isThereNewData() )

if __name__ == '__main__':
    unittest.main()