import unittest


class TestTemplateObject(unittest.TestCase):
    def test_object(self):
        import msmodel

        t = msmodel.TemplateObject()
        self.assertTrue(isinstance(t, msmodel.TemplateObject))
        return


if __name__ == '__main__':
    unittest.main()
