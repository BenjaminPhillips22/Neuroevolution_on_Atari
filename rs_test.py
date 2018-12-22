import unittest


class RandomSearchTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_the_test(self):
        '''
        testing the tester
        '''
        self.assertEqual(4*4, 16, 'something went wrong')

    def test_the_tester(self):
        '''
        Test if imported packages load
        '''
        import json
        import cv2
        import time
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        import torch
        import torch.nn as nn
        from torch.autograd import Variable
        import torch.nn.functional as F

        import gym
        from gym import wrappers

    def test_random(self):
        """
        check that we get the expected random number
        """
        import random
        random.seed(111)
        self.assertEqual(random.randint(0, 10000), 3485)

    def test_torch_random(self):
        """
        Check torch gives us the expected random number
        """
        import torch
        import numpy as np
        torch.manual_seed(111)
        a = torch.rand(1)
        self.assertTrue(np.abs(a.numpy()[0] - 0.7155659) < 0.001)

    # def test_a1(self):
    #     """
    #     test if a small network gives us what we expect
    #     """
    #     import rs_tests.my_test_a1 as a1
    #     import numpy as np
    #     reward, frames = a1.main()
    #     self.assertEqual(reward, 80)
    #     # small variation in when done becomes True
    #     j = np.abs(frames - 1604)
    #     if j > 200:
    #         print('a1 frames = ' + str(frames))
    #     self.assertTrue(j < 200)

    # def test_a2(self):
    #     """
    #     test if a small network gives us what we expect
    #     when monitoring is True
    #     """
    #     import rs_tests.my_test_a2 as a2
    #     import numpy as np
    #     reward, frames = a2.main()
    #     self.assertEqual(reward, 80)
    #     # small variation in when done becomes True
    #     j = np.abs(frames - 1604)
    #     if j > 200:
    #         print('a2 frames = ' + str(frames))
    #     self.assertTrue(j < 200)

    def test_b1(self):
        """
        check with a big model if the reward is what we expect
        """
        import rs_tests.my_test_b1 as b1
        import numpy as np
        reward, frames = b1.main()
        # self.assertEqual(reward, 0.0)
        # small variation in when done becomes True
        j = np.abs(frames - 14396)
        if j > 200:
            print('b1 frames = ' + str(frames))
        self.assertTrue(j < 200)

    # def test_b2(self):
    #     """
    #     check with a big model if the reward is what we expect
    #     when monitoring
    #     """
    #     import rs_tests.my_test_b2 as b2
    #     import numpy as np
    #     reward, frames = b2.main()
    #     self.assertEqual(reward, 0.0)
    #     # small variation in when done becomes True
    #     j = np.abs(frames - 14396)
    #     if j > 200:
    #         print('b2 frames = ' + str(frames))
    #     self.assertTrue(j < 200)


if __name__ == '__main__':
    unittest.main()
