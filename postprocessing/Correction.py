import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as cs
from os import listdir
import csv
import tensorflow as tf
from spell import Spell
import re
from nltk.corpus import reuters
from nltk import bigrams, trigrams
from collections import Counter, defaultdict

r = csv.reader(open('Prediction.txt','r'),delimiter=',')  # output from LipType #
r =  list(r)
r = np.array(r)
data =  r.astype(np.float)

r1 = csv.reader(open('labels.txt','r'),delimiter=',')    # True labels #
r1 =  list(r1)
r1 = np.array(r1)
lab =  r1.astype(np.float)

x = tf.placeholder(tf.float64, shape=[None, 28])
x1 = tf.placeholder(tf.float64, shape=[None, 28])

W_outlabs = tf.get_variable("weight_outlabs",[28,128],dtype =  tf.float64, initializer = tf.random_normal_initializer(0,0.01))
b_outlabs = tf.get_variable("bias_outlabs",[128], dtype =  tf.float64,initializer = tf.constant_initializer(0.0))
outlabs2 = tf.tanh(tf.add(tf.matmul(x,W_outlabs),b_outlabs))
print ("outlabs2", outlabs2.get_shape())

W_outlabs11 = tf.get_variable("weight_outlabs1",[128,64],dtype =  tf.float64, initializer = tf.random_normal_initializer(0,0.01))
b_outlabs11 = tf.get_variable("bias_outlabs1",[64], dtype =  tf.float64,initializer = tf.constant_initializer(0.0))
outlabs21 = tf.tanh(tf.add(tf.matmul(outlabs2,W_outlabs11),b_outlabs11))

W_outlabs12 = tf.get_variable("weight_outlabs2",[64,32],dtype =  tf.float64, initializer = tf.random_normal_initializer(0,0.01))
b_outlabs12 = tf.get_variable("bias_outlabs2",[32], dtype =  tf.float64,initializer = tf.constant_initializer(0.0))
outlabs22 = tf.tanh(tf.add(tf.matmul(outlabs21,W_outlabs12),b_outlabs12))

W_outlabs1 = tf.get_variable("weight_outlabs12",[32,64],dtype =  tf.float64, initializer = tf.random_normal_initializer(0,0.01))
b_outlabs1 = tf.get_variable("bias_outlabs12",[64], dtype =  tf.float64,initializer = tf.constant_initializer(0.0))
outlabs3 = tf.tanh(tf.add(tf.matmul(outlabs22,W_outlabs1),b_outlabs1))

W_outlabs1 = tf.get_variable("weight_outlabs12",[64,128],dtype =  tf.float64, initializer = tf.random_normal_initializer(0,0.01))
b_outlabs1 = tf.get_variable("bias_outlabs12",[128], dtype =  tf.float64,initializer = tf.constant_initializer(0.0))
outlabs31 = tf.tanh(tf.add(tf.matmul(outlabs3,W_outlabs1),b_outlabs1))

W_outlabs1 = tf.get_variable("weight_outlabs12",[128,28],dtype =  tf.float64, initializer = tf.random_normal_initializer(0,0.01))
b_outlabs1 = tf.get_variable("bias_outlabs12",[28], dtype =  tf.float64,initializer = tf.constant_initializer(0.0))
outlabs32 = tf.tanh(tf.add(tf.matmul(outlabs31,W_outlabs1),b_outlabs1))
print ("outlabs32", outlabs32.get_shape())

cost = tf.nn.softmax_cross_entropy_with_logits(logits = outlabs32, labels = x1)
train_op =  tf.train.AdamOptimizer(0.001).minimize(cost)
#train_op = tf.train.AdamOptimizer(0.01).minimize(cost)
out_labss1 = tf.argmax(outlabs3,1)
labb1 = tf.argmax(x1,1)
##########crossentropyloss#########
equal = tf.reduce_sum(tf.to_float(tf.equal(out_labss1, labb1)),name = 'ToFloat')
not_equal = tf.reduce_sum(tf.to_float(tf.not_equal(out_labss1, labb1)),name = 'ToFloat')
accuracy = (equal)/(equal+not_equal)
epochs = 50
with tf.Session() as s:
	s.run(tf.initialize_all_variables())
	for j in range(epochs):
		accur = 0
		for i in range(30):
			ipt1=  data[i]
			ipt1 = np.reshape(ipt1,(1,28))
			ipt2 = lab[i]
			ipt2 = np.reshape(ipt2,(1,28))
			_,o1,acc,eq,neq,ll1 = s.run([train_op,out_labss1,accuracy,equal,not_equal,labb1], feed_dict={x:ipt1, x1:ipt2})
			f = open( 'DDA_prediction.txt', 'a' )
			f.write( 'dict = ' + repr(o1) + '\n' )
			f.close()
			accur  = accur + acc
		print "accuracy:", accur/30

############Language model##############

model = defaultdict(lambda: defaultdict(lambda: 0))
model1 = defaultdict(lambda: defaultdict(lambda: 0))
with open('LM-corpus.txt') as f:           # Librispeech Corpus #
        lines = f.readlines()
for i in lines:
        # wordList = re.sub("[^\w]", " ",  i).split()
        for w1, w2, w3 in trigrams(re.sub("[^\w]", " ",  i).split(), pad_right=True, pad_left=True):
                model[(w1, w2)][w3] += 1
        for p1, p2, p3 in trigrams(re.sub("[^\w]", " ",  i).split(), pad_right=True, pad_left=True):
                model1[(p3, p2)][p1] += 1
 
# Let's transform the counts to probabilities
for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count

for p3_p2 in model1:
    total_count1 = float(sum(model1[p3_p2].values()))
    for p1 in model1[p3_p2]:
        model1[p3_p2][p1] /= total_count1
PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'..','..','labels.txt')
sen = csv.reader(open('DDA_prediction.txt','r'),delimiter=',')
sen =  list(sen)
sen = np.array(sen)
sen =  sen.astype(np.float)
spell = Spell(path=PREDICT_DICTIONARY, sen)# print(s)
final1=''
for word in range(len(spell.split())-1):
        dic = (dict(model[s[0], s[1]]))
        print("predicted next word: ",dic)
        v = list(dic.values())
        v1 = list(dic.keys())
        if(all(x==v[0] for x in v))==True and len(dic)>1:
                dic1 = (dict(model1[v1[0], s[1]]))
                dic2 = (dict(model1[v1[1], s[1]]))
                Keymax1 = dic1.get(s[0])
                Keymax2 = dic2.get(s[0])
                if (Keymax1>Keymax2):
                        Keymax = v1[0]
                else:
                        Keymax = v1[1]
        else:
                Keymax = max(dic, key=dic.get)
        if word==0:
                s = " ".join(s)
                final1 = final1+' '+s+' '+Keymax
        else:
                final1 = final1 +' '+Keymax 
        s = final1.split()[-2:]
for w in s:        
        if P[w] < 0.7:
	new = known(edits2(word))
        s.replace(w, new)
print ("output", s)

def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)
def known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.dictionary)

def edits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))