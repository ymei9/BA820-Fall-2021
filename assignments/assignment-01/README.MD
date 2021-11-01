# Assignment 1

The file `forums.pkl` is the dataset for your Assignment.

- The Assignment instructions can be found on Questrom Tools under Tests/Quizzes
- This is __individual assignment__.  You should not work or discuss this with anyone in the program.  
  
This file can be easily read into python via

```
import pandas as pd
forums = pd.read_pickle("forums.pkl")
```

# Tips and Tricks

You may get an error, depending on your approach.  In the event you get the following error:

> `RecursionError: maximum recursion depth exceeded while calling a Python object`

You should be able to resolve this issue with 

```
import sys
sys.setrecursionlimit(10000)
```

> Above should resolve the issue, and was tested on Colab.

