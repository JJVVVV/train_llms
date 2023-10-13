#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ngram util
"""
from __future__ import print_function
from __future__ import unicode_literals

import re
import collections


class _NaiveCnCharTokenizer(object):
    """cn char tokenizer re-impl
    inspried by 
    https://stackoverflow.com/questions/30425877/\
    how-to-not-split-english-into-separate-letters-in-the-stanford-chinese-parser
    ===
    because IT is hard to get all Chinese codecs, so we have to find the ASCI parts.
    and the extra parts are seen as Chinese. it it not OK. but I have to.
    https://github.com/tsroten/zhon seems to be a nice repo to process Chinese.
    But we have to considering the C++ impl.
    SO LET it be. 
    """
    # https://stackoverflow.com/questions/\
    # 1366068/whats-the-complete-range-for-chinese-characters-in-unicode
    # _CN_CHAR = (ur"[\u3000-\u303f\u3400-\u4DBF\u4e00-\uffff]")
    # # {cn}\s+{cn} must be the front of {\n}!! it stands for the match order!!
    # CN_PART_RE = re.compile(ur"(?:{cn}\s+{cn}|{cn})+".format(cn=_CN_CHAR))
    # ALL_SPACE_RE = re.compile(ur"^\s+$")

    EN_PART_RE = re.compile("[\s\u0021-\u007f]+")

    @classmethod
    def tokenize(cls, s, lowercase=True):
        """tokenize for str
        Parameters
        -----------
        s: unicode
        """
        token_list = []
        part_spos = 0
        for en_match_obj in cls.EN_PART_RE.finditer(s):
            en_part_spos, en_part_epos = en_match_obj.span()
            cn_part_str = s[part_spos: en_part_spos]
            en_part_str = s[en_part_spos: en_part_epos]
            cn_token_list = list(cn_part_str)
            # naively use whitespace as non-cn separator
            en_token_list = en_part_str.split()
            
            token_list.extend(cn_token_list)
            token_list.extend(en_token_list)
            part_spos = en_part_epos
        cn_part_str = s[part_spos:]
        token_list.extend(list(cn_part_str))
        # remove empty character('')
        return [_s for _s in token_list if _s]
    

get_cn_char_unigram = _NaiveCnCharTokenizer.tokenize


def gen_ngram(unigram_list, n, padding=False):
    """generate ngrams

    Paragrams
    ============
    padding: if padding, use <sos> and <eos>.
        example:
            seq = ABC
            padding = Ture:
                bigram: <sos>-A, A-B, B-C, C-<eos>
            padding = False:
                bigram: A-B, B-C
        EDGE example:
            SEQ = None
            padding = True:
                bigram: <sos>-<eos>
            padding = False:
                bigram: None (realy return is a empty list)
    SPECIAL:
        SEQ = []
        padding = True:
            unigram: [] # not <sos>, <eos> !! we accept this result!!
        padding = False:
            unigram: []
    """
    if not isinstance(n, int) or n < 1:
        raise Exception("un-expected value: {}".format(n))
    if n == 1:
        return unigram_list
    SOS = u"<sos>"
    EOS = u"<eos>"
    JOIN_CHAR = u"-"
    ngram_queue = collections.deque(maxlen=n)
    unigram_sz = len(unigram_list)
    if padding:
        ngram_queue.extend([SOS] * (n - 1))
        start_pos = 0
    else:
        if unigram_sz < n:
            return [] # no n-gram
        ngram_queue.extend(unigram_list[: n - 1])
        start_pos = n - 1
    ngram_list = []
    for front_idx in range(start_pos, unigram_sz):
        unigram = unigram_list[front_idx]
        ngram_queue.append(unigram)
        joint_val = JOIN_CHAR.join(ngram_queue)
        ngram_list.append(joint_val)
    if padding:
        for i in range(n - 1):
            ngram_queue.append(EOS)
            joint_val = JOIN_CHAR.join(ngram_queue)
            ngram_list.append(joint_val)
    return ngram_list


def _test_gen_ngram():
    """test gen ngram
    """
    from pprint import pprint
    unigram_list = ["a", "b", "c", "d", "e"]
    pprint(gen_ngram(unigram_list, 2, True))
    pprint(gen_ngram(unigram_list, 2, False))

    unigram_list = ["a", "b"]
    pprint(gen_ngram(unigram_list, 2, True))
    pprint(gen_ngram(unigram_list, 2, False))

    unigram_list = ["a",]
    pprint(gen_ngram(unigram_list, 2, True))
    pprint(gen_ngram(unigram_list, 2, False))

    unigram_list = []
    pprint(gen_ngram(unigram_list, 2, True))
    pprint(gen_ngram(unigram_list, 2, False))
    
    unigram_list = ["a",]
    pprint(gen_ngram(unigram_list, 1, True))
    pprint(gen_ngram(unigram_list, 1, False))

        
    unigram_list = []
    pprint(gen_ngram(unigram_list, 1, True))
    pprint(gen_ngram(unigram_list, 1, False))


def _test_get_cn_char_unigram():
    """ test get-cn-char-unigram
    """
    s = "今天temperature是-0.5度。"
    unigrams = get_cn_char_unigram(s)
    for w in unigrams:
        print(w)

    s = "OH! 今 天 -15（16.8 度吧"
    unigrams = get_cn_char_unigram(s)
    for w in unigrams:
        print(w)
    print("========")
    s = "      https://hahah.com "
    unigrams = get_cn_char_unigram(s)
    for w in unigrams:
        print(w)

if __name__ == "__main__":
    _test_get_cn_char_unigram()


