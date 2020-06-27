import unittest


class TestTemplateObject(unittest.TestCase):
    def test_object(self):
        import patient

        t = patient.TemplateObject()
        self.assertTrue(isinstance(t, patient.TemplateObject))
        return


if __name__ == '__main__':
    unittest.main()
