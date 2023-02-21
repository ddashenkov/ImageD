import unittest
import dataset


class DatasetTest(unittest.TestCase):

    def test_read_train(self):
        ds = dataset.read_train()
        self.assertEqual(ds.shape, (6622219, 2))
        self.assertRegex(ds[0, 0], "[0-9a-f]{16}")
        self.assertRegex(ds[0, 1], "/m/\\w+")


if __name__ == '__main__':
    unittest.main()
