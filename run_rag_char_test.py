import os
os.environ.pop('GOOGLE_API_KEY', None)
import Rag_Char as m
print('MODULE IMPORTED')
print('answer:', m.ask('What are the main issues patients mention?'))
