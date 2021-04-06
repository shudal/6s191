from lab1_2_2 import *

if __name__ == '__main__':
    char2idx={}
    char2idx['a']=1
    char2idx['b']=2

    tests="abaa"
    s=vectorize_string(tests,char2idx)
    print(s)