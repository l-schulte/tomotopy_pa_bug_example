import tomotopy as tp
import json

mdl = tp.PAModel(k1=150, k2=250, seed=123)

documents = json.load(open('export.json', 'r'))

for doc in documents:
    mdl.add_doc(doc)

for i in range(0, 100, 10):
    mdl.train(10)
    print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

print('PA ll per word  \t{}'.format(mdl.ll_per_word))

mdl.save('pa.mdl')
