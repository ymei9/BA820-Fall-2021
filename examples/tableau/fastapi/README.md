# About

Tableau has the ability to call out to analytical backends powered by R or python.  I am going to use the newly launched `fastapitableau` package from the Rstudio folks.

https://github.com/rstudio/fastapitableau

The app can be run with:

```
uvicorn main:app --reload
```

as the service is found in `main.py` but assumes that the command is run from the subdir as the current working directory.


## Usage

After the API is running:

- `localhost:8000` will bring up some docs that the RStudio folks wrapped to help with the usage
-  RStudio assumes that the elements are `typed`, which you can see in the script.
- `localhost:8000/docs` will bring up the FastAPI version of the docs.  This is interactive!

From within Tableau (after FastAPI is running):

- To tell Tableau that there is Analytics Server, Help > Settings and Performance > Manage Analytics Extension Connection ...
    - Select Analytics Extension
    - localhost:8000
    - test the connection and then click ok
- Disable Aggregation by default via Analysis > Aggregate Measures
- When creating a function to use, you may need to wrap the Tableau field with `ATTR([FIELD_HERE])`
    - The RStudio folks suggest that this is required because of the disabling of Aggregate Measures

Assuming that we are using the Bruins Twitter dataset from Big Query (yes, we can connect to BQ via Tableau!), I would use the sentiment score API via:

```
SCRIPT_REAL("/sentiment", ATTR([Source]))
```




