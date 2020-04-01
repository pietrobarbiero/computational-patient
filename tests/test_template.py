import unittest


class TestTemplateObject(unittest.TestCase):
    def test_object(self):
        import pkpd

        t = pkpd.TemplateObject()
        self.assertTrue(isinstance(t, pkpd.TemplateObject))
        return


if __name__ == '__main__':
    unittest.main()
