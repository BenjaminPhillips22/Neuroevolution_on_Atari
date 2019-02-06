import unittest
import torch

import base_model


class GATests(unittest.TestCase):

    def setUp(self):
        pass

    def test_the_test(self):
        '''
        testing the tester
        '''
        self.assertEqual(4*4, 16, 'something went wrong')

    def test_seed_dict(self):
        '''
        test that BigModel weights are the same and different
        as I would expect.
        '''
        a_seed_dict = {'conv1.weight': [248528185],
                       'conv2.weight': [1877002014],
                       'conv3.weight': [1679996776],
                       'dense.weight': [919992575],
                       'out.weight': [1904593925]}

        b_seed_dict = {'conv1.weight': [248528185],
                       'conv2.weight': [1877002014, 123],
                       'conv3.weight': [1679996776],
                       'dense.weight': [919992575],
                       'out.weight': [1904593925]}

        m1 = base_model.BigModel(a_seed_dict)
        m2 = base_model.BigModel(b_seed_dict)

        a_tens = []
        for name, tensor in m1.named_parameters():
            a_tens.append(tensor)

        b_tens = []
        for name, tensor in m2.named_parameters():
            b_tens.append(tensor)
            
        result = []    
        for i in range(len(a_tens)):
            result.append(torch.all(torch.eq(a_tens[i], b_tens[i])).data)

        self.assertEqual(result, [1, 1, 0, 1, 1, 1, 1, 1, 1, 1])


if __name__ == '__main__':
    unittest.main()
