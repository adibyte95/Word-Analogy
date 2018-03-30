
# coding: utf-8

# In[1]:


# importing dependencies
import numpy as np


# In[2]:


# for loading the glove word embedding matrix values
def load_glove_vectors(glove_file):
    with open(glove_file, 'r', encoding="utf-8") as file:
        # unique words
        words = set()
        word_to_vec = {}
        # each line starts with a word then the values for the different features
        for line in file:
            line = line.strip().split()
            # take the word 
            curr_word = line[0]
            words.add(curr_word)
            # rest of the features for the word
            word_to_vec[curr_word] = np.array(line[1:], dtype=np.float64)
            
    return words, word_to_vec


# In[3]:


words, word_to_vec = load_glove_vectors('data/glove.6B.300d.txt')


# In[4]:


def find_cosine_similarity(u, v):
    distance = 0.0
    
    # find the dot product between u and v 
    dot = np.dot(u,v)
    # find the L2 norm of u 
    norm_u = np.sqrt(np.sum(u**2))
    # Compute the L2 norm of v
    norm_v = np.sqrt(np.sum(v**2))
    # Compute the cosine similarity
    distance = dot/(norm_u)/norm_v
    
    return distance


# In[5]:


father = word_to_vec["father"]
mother = word_to_vec["mother"]
king = word_to_vec["king"]
queen = word_to_vec["queen"]
bat = word_to_vec["bat"]
crow = word_to_vec["crow"]
india = word_to_vec["india"]
italy = word_to_vec["italy"]
delhi = word_to_vec["delhi"]
rome = word_to_vec["rome"]

print("cosine_similarity(king, queen) = ", find_cosine_similarity(king, queen))
print("cosine_similarity(father, mother) = ", find_cosine_similarity(father, mother))
print("cosine_similarity(king - queen, father - mother) = ",find_cosine_similarity(king - queen, father - mother))
print("cosine_similarity(bat, crow) = ",find_cosine_similarity(bat, crow))
print("cosine_similarity(india - delhi, rome - italy) = ",find_cosine_similarity(india - delhi, rome - italy))


# In[6]:


# does the Word analogy task: a is to b as c is to ____
def find_analogy(word_a, word_b, word_c, word_to_vec):
    # convert words to lower case
    word_a = word_a.lower()
    word_b = word_b.lower()
    word_c = word_c.lower()
    
    
    # find the word embeddings for word_a, word_b, word_c
    e_a, e_b, e_c = word_to_vec[word_a], word_to_vec[word_b], word_to_vec[word_c]
    
    words = word_to_vec.keys()
    max_cosine_sim = -999              
    best_word = None                  

    # search for word_d in the whole word vector set
    for w in words:        
        # ignore input words
        if w in [word_a, word_b, word_c] :
            continue

        # Compute cosine similarity between the vectors u and v
        #u:(e_b - e_a) 
        #v:((w's vector representation) - e_c)
        cosine_sim = find_cosine_similarity(e_b - e_a, word_to_vec[w] - e_c)
        
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            # update word_d
            best_word = w
        
    return best_word


# In[7]:


examples = [('india', 'delhi', 'japan'), ('tall', 'taller', 'large')]
for example in examples:
    print ('{} -> {} :: {} -> {}'.format( *example, find_analogy(*example, word_to_vec)))


# In[8]:


# for taking input from the user and doing word analogy task on that
def take_input():
    print('a --> b :: c --> d')
    print('Enter a, b, c words separated by space')
    words = input().split(' ')
    
    best_pick = find_analogy(*words, word_to_vec)
    print ('{} -> {} :: {} -> {}'.format( *words, best_pick))
    print('Best pick: ' + best_pick)


# In[9]:


take_input()


# In[14]:


take_input()


# In[15]:


take_input()

