import re
import numpy as np
from python_speech_features import mfcc
from python_speech_features import delta
import scipy.io.wavfile as wav

def read_transcript(file_path):
    with open(file_path, "r") as file:
        contents = file.readlines()
    contents = [ele.strip().split(" ")[-1] for ele in contents]
    contents = " ".join(contents)
    return contents

def clean_transcripts(transcript_list):
    for i in range(len(transcript_list)):
        transcript_list[i] = transcript_list[i].lower()
        transcript_list[i] = re.sub('[^A-Za-z ]+', '', transcript_list[i])
    return transcript_list

def get_dict(dataset):
    vocab = set(dataset)
    vocab = list(vocab)
    vocab.sort()
    indices = np.arange(len(vocab))
    transcript_dict = dict(zip(vocab, indices))
    return transcript_dict

def sentence_to_indices(sentence, transcript_dict):
    return [transcript_dict[character] for character in sentence]

def indices_to_sentence(indices, transcript_dict):
    inverted_dict = dict([[v,k] for k,v in transcript_dict.items()])
    return "".join([inverted_dict[index] for index in indices])

def build_transcript_dataset(transcript_list):
    numpy_tr_list = np.asanyarray(transcript_list)
    numpy_tr_list = numpy_tr_list.flatten()
    numpy_tr_string = " ".join(transcript_list)
    transcript_dict = get_dict(numpy_tr_string)
    character_indexes = []
    for transcript in transcript_list:
        character_indexes.append(sentence_to_indices(transcript, transcript_dict))
    return character_indexes, transcript_dict

def get_mfcc(wav_file, deltas = 1, context = 2):
    (rate,sig) = wav.read(wav_file)
    mfcc_feat = mfcc(sig,rate)
    mfcc_orig = mfcc_feat.copy()
    if(deltas):
        for i in range(deltas):
            mfcc_delta = delta(mfcc_orig, context)
            mfcc_feat = np.hstack((mfcc_feat, mfcc_delta))
        mfcc_orig = mfcc_feat
    return mfcc_orig

def get_mfcc_for_wavs(wav_list, deltas = 1, context = 2):
    features = []
    list_size = len(wav_list)
    for i in range(list_size):
        features.append(get_mfcc(wav_list[i], deltas = deltas, context = context))
        percentage_completion = (100./list_size*i)
        if(percentage_completion % 10 == 0):
            print("{0}% mfcc computation completed".format(percentage_completion))
    features = np.asanyarray(features)
    return features
