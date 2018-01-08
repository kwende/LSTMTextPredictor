import numpy as n
import re
import random

def read_training_data(path):
    ret = []
    lookup = {}

    text = ""
    with open(path, "r") as file:
        allLines = file.readlines()

        for line in allLines:
            if line.startswith("\n"):
                if len(text) > 0:
                    ret.append(text.strip())
                text = ""
            else:
                if text.endswith(" "):
                    text = text + line
                else:
                    text = text + line + " "

    if len(text) > 0:
        ret.append(text.strip())

    return ret

def build_word_dictionary(allData):

    dict = {}

    key = 0
    for data in allData:
        chunks = re.findall(r"[\w']+|[.,!?;]", data)
        for chunk in chunks:
            if not chunk in dict:
                dict[chunk] = key
                key = key + 1
    
    return dict

def get_word_sequence(data, numWords):

    chunks = re.findall(r"[\w']+|[.,!?;]", data)
    numChunks = len(chunks)

    if (numChunks + 1) >= numWords:
        startPosition = random.randint(0, len(chunks) - (numWords + 1))

        sequence = chunks[startPosition:(startPosition + numWords)]
        next = chunks[startPosition + numWords]

        return (sequence, next)
    else:
        return (chunks[0:numChunks - 2], chunks[numChunks] - 1)

def get_word_sequence_batch(data, batch_size, sequence_length):

    batch = []

    for i in range(0, len(data)):
        index = random.randint(0, len(data) - 1)
        batch.append(get_word_sequence(data[i], sequence_length))

    return batch

def convert_batch_to_numbers(batch, dict):

    nBatch = []

    for b in batch:
        nBatch.append(([dict[a] for a in b[0]], dict[b[1]]))

    return nBatch