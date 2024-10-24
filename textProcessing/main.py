import loader, preProcess, naive

print("loading datasets...")
x_raw, y = loader.getData("train.txt")
x_raw_test, y_test = loader.getData("test.txt")


print("Generating bag of words...")
bagOfWords = {}

for val in x_raw:
    cleaned = preProcess.cleanLinks(val)
    cleaned = preProcess.cleanHearts(cleaned)
    cleaned = preProcess.cleanSaddies(cleaned)
    cleaned = preProcess.cleanSmies(cleaned)

    bagOfWords = preProcess.stem(cleaned, bagOfWords)

sorted_items = sorted(bagOfWords.items(), key=lambda item: item[1])

first_keys = [key for key, value in sorted_items[:60]]
last_keys = [key for key, value in sorted_items[-55:]]

bag = last_keys + first_keys

print(len(last_keys), last_keys[0])

print("preparing X and X_test vectors...")

X = [preProcess.getInfo(bag, i) for i in x_raw]

X_test = [preProcess.getInfo(bag, i) for i in x_raw_test]

print("Training and predicting...")
naive.trainAndTest(X, y, X_test, y_test)