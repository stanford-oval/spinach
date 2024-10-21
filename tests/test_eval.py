import unittest
import sys, os
import math
current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_script_path)
sys.path.append(parent_directory)

from eval import f1

class Testf1(unittest.TestCase):
    def assertEqualRound(self, a, b):
        """
        Scipy processes floats a little bit differently from the default python processing here.
        Thus using a rounding mechanism to check results
        """
        factor = 10 ** 10
        a = math.ceil(a * factor) / factor
        b = math.ceil(b * factor) / factor
        return self.assertEqual(a,b)
    
    def test_f1_simple_em(self):
        predicted = [{'item': {'type': 'uri', 'value': 'a'}, 'itemLabel': {'type': 'literal', 'value': 'Northern Tsou', 'xml:lang': 'en'}}]
        gold = [{'item': {'type': 'uri', 'value': 'a'}, 'itemLabel': {'type': 'literal', 'value': 'Northern Tsou', 'xml:lang': 'en'}}]
        self.assertEqualRound(f1(predicted, gold), 1)

    def test_f1_simple_no_label(self):
        predicted = [{'item': {'type': 'uri', 'value': 'a'}}]
        gold = [{'item': {'type': 'uri', 'value': 'a'}, 'itemLabel': {'type': 'literal', 'value': 'Northern Tsou', 'xml:lang': 'en'}}]
        self.assertEqualRound(f1(predicted, gold), 1)

    def test_f1_2_em(self):
        predicted = [{'item': {'type': 'uri', 'value': 'a'}}, {'item': {'type': 'uri', 'value': 'a'}}]
        gold = [{'item': {'type': 'uri', 'value': 'a'}}, {'item': {'type': 'uri', 'value': 'a'}}]
        self.assertEqualRound(f1(predicted, gold), 1)
        
    def test_f1_2_e(self):
        predicted = [{'item': {'type': 'uri', 'value': 'a'}}, {'item': {'type': 'uri', 'value': 'a'}}]
        gold = [{'item': {'type': 'uri', 'value': 'a'}}, {'item': {'type': 'uri', 'value': 'a'}}]
        self.assertEqualRound(f1(predicted, gold), 1)
        
    def test_f1_2_em_2(self):
        predicted = [{'item': {'type': 'uri', 'value': 'a'}}, {'item': {'type': 'uri', 'value': 'b'}}]
        gold = [{'item': {'type': 'uri', 'value': 'b'}}, {'item': {'type': 'uri', 'value': 'a'}}]
        self.assertEqualRound(f1(predicted, gold), 1)

    def test_f1_5_items_fp(self):
        predicted = [{'item': {'type': 'uri', 'value': 'a'}}, {'item': {'type': 'uri', 'value': 'a'}}, {'item': {'type': 'uri', 'value': 'c'}}, {'item': {'type': 'uri', 'value': 'd'}}, {'item': {'type': 'uri', 'value': 'f'}}]
        gold = [{'item': {'type': 'uri', 'value': 'a'}}]
        self.assertEqualRound(f1(predicted, gold), 2 / (2 + 0 + 4))
        
    def test_f1_5_items_fn(self):
        predicted = [{'item': {'type': 'uri', 'value': 'a'}}]
        gold = [{'item': {'type': 'uri', 'value': 'a'}}, {'item': {'type': 'uri', 'value': 'a'}}, {'item': {'type': 'uri', 'value': 'c'}}, {'item': {'type': 'uri', 'value': 'd'}}, {'item': {'type': 'uri', 'value': 'f'}}]
        self.assertEqualRound(f1(predicted, gold), 2 / (2 + 0 + 4))
        
    def test_f1_5_items_2fn(self):
        predicted = [{'item': {'type': 'uri', 'value': 'a'}}, {'item': {'type': 'uri', 'value': 'c'}}]
        gold = [{'item': {'type': 'uri', 'value': 'a'}}, {'item': {'type': 'uri', 'value': 'a'}}, {'item': {'type': 'uri', 'value': 'c'}}, {'item': {'type': 'uri', 'value': 'd'}}, {'item': {'type': 'uri', 'value': 'f'}}]
        self.assertEqualRound(f1(predicted, gold), 2 * 2 / (2 * 2 + 3))

    def test_f1_5_items_2fn_1fp(self):
        predicted = [{'item': {'type': 'uri', 'value': 'a'}}, {'item': {'type': 'uri', 'value': 'c'}}, {'item': {'type': 'uri', 'value': 'surprise'}}]
        gold = [{'item': {'type': 'uri', 'value': 'a'}}, {'item': {'type': 'uri', 'value': 'a'}}, {'item': {'type': 'uri', 'value': 'c'}}, {'item': {'type': 'uri', 'value': 'd'}}, {'item': {'type': 'uri', 'value': 'f'}}]
        self.assertEqualRound(f1(predicted, gold), 2 * 2 / (2 * 2 + 1 + 3))
        
    def test_f1_2_em_2_all_half(self):
        predicted = [{'item': {'type': 'uri', 'value': 'a'}}, {'item': {'type': 'uri', 'value': 'b'}}]
        gold = [{'item': {'type': 'uri', 'value': 'b'}, 'surprise': {'type': 'uri', 'value': 'c'}}, {'item': {'type': 'uri', 'value': 'a'}, 'surprise': {'type': 'uri', 'value': 'c'}}]
        self.assertEqualRound(f1(predicted, gold), 2/(2 + 1))
        
    def test_f1_2_em_2_all_half_w_2fp(self):
        predicted = [{'item': {'type': 'uri', 'value': 'a'}}, {'item': {'type': 'uri', 'value': 'b'}}, {'item': {'type': 'uri', 'value': 'surprise'}}, {'item': {'type': 'uri', 'value': 'surprise_2'}}]
        gold = [{'item': {'type': 'uri', 'value': 'b'}, 'surprise': {'type': 'uri', 'value': 'c'}}, {'item': {'type': 'uri', 'value': 'a'}, 'surprise': {'type': 'uri', 'value': 'c'}}]
        self.assertEqualRound(f1(predicted, gold), 2/(2 + 1 + 2))
        
    def test_f1_2_em_2_all_one_third_w_2fp(self):
        predicted = [{'item': {'type': 'uri', 'value': 'a'}}, {'item': {'type': 'uri', 'value': 'b'}}, {'item': {'type': 'uri', 'value': 'surprise'}}, {'item': {'type': 'uri', 'value': 'surprise_2'}}]
        gold = [{'item': {'type': 'uri', 'value': 'b'}, 'surprise': {'type': 'uri', 'value': 'c'}, 'surprise_2': {'type': 'uri', 'value': 'c'}}, {'item': {'type': 'uri', 'value': 'a'}, 'surprise': {'type': 'uri', 'value': 'c'}, 'surprise_2': {'type': 'uri', 'value': 'c'}}]
        self.assertEqualRound(f1(predicted, gold), ((1/3) * 2 * 2)/((1/3) * 2 * 2 + (2/3) * 2 + 2))

    def test_f1_2_em_2_all_two_third_w_2fp(self):
        predicted = [{'item': {'type': 'uri', 'value': 'b'}, 'surprise': {'type': 'uri', 'value': 'c'}}, {'item': {'type': 'uri', 'value': 'a'}, 'surprise': {'type': 'uri', 'value': 'c'}}, {'item': {'type': 'uri', 'value': 'surprise'}, 'surprise': {'type': 'uri', 'value': 'c'}}, {'item': {'type': 'uri', 'value': 'surprise_2'}, 'surprise': {'type': 'uri', 'value': 'c'}}]
        gold = [{'item': {'type': 'uri', 'value': 'b'}, 'surprise': {'type': 'uri', 'value': 'c'}, 'surprise_2': {'type': 'uri', 'value': 'abc'}}, {'item': {'type': 'uri', 'value': 'a'}, 'surprise': {'type': 'uri', 'value': 'c'}, 'surprise_2': {'type': 'uri', 'value': 'abc'}}]
        self.assertEqualRound(f1(predicted, gold), ((2/3) * 2 * 2)/((2/3) * 2 * 2 + (1/3) * 2 + 2))
        
    def test_f1_2_em_2_all_two_third_w_2fp_duplicate_gold_should_not_matter(self):
        predicted = [{'item': {'type': 'uri', 'value': 'b'}, 'surprise': {'type': 'uri', 'value': 'c'}}, {'item': {'type': 'uri', 'value': 'a'}, 'surprise': {'type': 'uri', 'value': 'c'}}, {'item': {'type': 'uri', 'value': 'surprise'}, 'surprise': {'type': 'uri', 'value': 'c'}}, {'item': {'type': 'uri', 'value': 'surprise_2'}, 'surprise': {'type': 'uri', 'value': 'c'}}]
        gold = [{'item': {'type': 'uri', 'value': 'b'}, 'surprise': {'type': 'uri', 'value': 'c'}, 'surprise_2': {'type': 'uri', 'value': 'c'}}, {'item': {'type': 'uri', 'value': 'a'}, 'surprise': {'type': 'uri', 'value': 'c'}, 'surprise_2': {'type': 'uri', 'value': 'c'}}]
        self.assertEqualRound(f1(predicted, gold), ((2/3) * 2 * 2)/((2/3) * 2 * 2 + (1/3) * 2 + 2))
    
if __name__ == '__main__':
    unittest.main()