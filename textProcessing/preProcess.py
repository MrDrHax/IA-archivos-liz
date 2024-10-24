import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')

linkRegex = r'https?:\/\/\S+\.\S+'


def cleanLinks(data: str) -> str:
    return re.sub(linkRegex, "URL", data)


def cleanSmies(data: str) -> str:
    return re.sub(r":[\)3]", "smile", data)


def cleanHearts(data: str) -> str:
    return re.sub(r"<3", "heart", data)


def cleanSaddies(data: str) -> str:
    return re.sub(r":[\(cC]", "sad", data)


def getSmileCount(data: str):
    return len(re.findall(r'smile', data))


def getSadCount(data: str):
    return len(re.findall(r'sad', data))


def getUrlCount(data: str):
    return len(re.findall(r'URL', data))


def getHeartCount(data: str):
    return len(re.findall(r'heart', data))


def countMayus(data: str) -> int:
    return len(re.findall(r'[A-Z]', data))


def countExclamation(data: str):
    return len(re.findall(r"[!$¡]", data))


def countQuestions(data: str):
    return len(re.findall(r"[\?-¿%]", data))


def charactersCount(data: str) -> int:
    return len(data)


def wordCount(data: str) -> int:
    return len(data.split(" "))


def stem(data: str, initialDict: dict[str, int] | None = None) -> dict[str, int]:
    tokenized = word_tokenize(data)

    toReturn = {} if initialDict == None else initialDict
    p = PorterStemmer()

    for word in tokenized:
        stemmed = p.stem(word)
        toReturn[stemmed] = toReturn.get(stemmed, 0) + 1

    return toReturn


def getInfo(bagOfWords: list[str], data: str) -> list[int]:
    # pre cleaning
    data = cleanLinks(data)
    data = cleanSmies(data)
    data = cleanHearts(data)
    data = cleanSaddies(data)

    # get all the juice
    mayus = countMayus(data)
    positive = countExclamation(data)
    negative = countQuestions(data)
    count = charactersCount(data)
    words = wordCount(data)
    positiveNegRatio = positive / negative
    wordsVsCount = words / count
    mayusPercent = mayus/count

    vector = [mayus, positive, negative, count, words, positiveNegRatio, wordsVsCount,
              mayusPercent, getSmileCount(data), getHeartCount(data), getSadCount(data), getUrlCount(data)]

    # clean and process to get bag of words
    tokenized = stem(data)

    for word in bagOfWords:
        vector.append(tokenized.get(word, 0))

    return vector
