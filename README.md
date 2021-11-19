# experiment-analysis
A package of functions for analyzing online experiments

Background on the need for this repo is [here](https://www.notion.so/faire/Re-usable-ranking-team-functions-for-Mode-s-Py-Notebook-7688e5d52b0849db9fabd99d5c80ba15).

While the data team largely relies on the Experiment pipeline and dashboard to analyze online experiments, many members are creating their own offline analyses in Mode's notebook. This can be error prone as it required copying and pasting very large analysis functions for every notebook.

The ideal state is to have shared functions for this work that the data team can import in Mode. Mode allows pip installable packages, but only for Public repos. This repo will be a small package of (peer-reviewed) stats functions maintained by the experimentation team on core data infra (Elizabeth is the lead there) that can be imported into Mode notebooks.

There is no current way in the Mode product to pip install private repos.
