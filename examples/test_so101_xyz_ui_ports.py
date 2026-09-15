import unittest

from so101_xyz_ui import validate_port


class PortTests(unittest.TestCase):
    def test_valid_ports(self):
        self.assertEqual(validate_port(" com6 "), "COM6")
        self.assertEqual(validate_port("COM12"), "COM12")

    def test_invalid_ports(self):
        for value in ("", "COM0", "COM-1", "COM6;echo", "ttyUSB0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_port(value)


if __name__ == "__main__":
    unittest.main()
