import pickle
def unpickle_iter(file):
    try:
        while True:
             yield pickle.load(file)
    except EOFError:
        raise StopIteration

with open('../WESAD/S2/S2.pkl', 'rb') as file:
    for item in unpickle_iter(file):
        print(item)
