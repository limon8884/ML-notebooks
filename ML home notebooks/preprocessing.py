from dataclasses import dataclass
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    
    source: List[str]
    target: List[str]
    


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    content = open(filename, 'r').read().replace('&', '&amp;')#.replace("[\u0000-\u0008,\u000B,\u000C,\u000E-\u001F]", "")#.translate(str.maketrans({'\u00ad':'', }))
    root = ET.fromstring(content)
    sentpairs = []
    labeledaligments = []
    for elem in root:
        sentpairs.append(SentencePair(elem.find('english').text.split(), elem.find('czech').text.split()))
        sure = []
        if elem.find('sure').text is not None:
            sure = [tuple(map(int, e.split('-'))) for e in elem.find('sure').text.split()]
        possible = []
        if elem.find('possible').text is not None:
            possible = [tuple(map(int, e.split('-'))) for e in elem.find('possible').text.split()]

        labeledaligments.append(LabeledAlignment(sure, possible))    
        
    return (sentpairs, labeledaligments)

#print(extract_sentences('D:/GitHub/ml-course-hse/2020-spring/homeworks-practice/homework-practice-09-em/project_syndicate_bacchetta1.wa'))
def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    token_to_index_for_source = {}
    token_to_index_for_target = {}
    ind_s = 0
    ind_t = 0
    for sp in sentence_pairs:
        for word in sp.source:
            if token_to_index_for_source.get(word) is None:
                token_to_index_for_source[word] = ind_s
                ind_s += 1
        for word in sp.target:
            if token_to_index_for_target.get(word) is None:
                token_to_index_for_target[word] = ind_t
                ind_t += 1
    if freq_cutoff is None:
        return (token_to_index_for_source, token_to_index_for_target)
    return (dict(sorted(token_to_index_for_source.items(), key=lambda item: item[1], reverse=True)[:freq_cutoff]), 
    dict(sorted(token_to_index_for_target.items(), key=lambda item: item[1], reverse=True)[:freq_cutoff])) 
    

def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    ans = []
    for sp in sentence_pairs:
        source_list = []
        target_list = []
        no_Nones_source = True
        for word in sp.source:
            if source_dict.get(word) is None:
                no_Nones_source = False
                break
            source_list.append(source_dict[word])
        no_Nones_target = True
        for word in sp.target:
            if target_dict.get(word) is None:
                no_Nones_target = False
                break
            target_list.append(target_dict[word])
        if no_Nones_source and no_Nones_target:
            ans.append(TokenizedSentencePair(np.array(source_list), np.array(target_list)))
    return ans

                
