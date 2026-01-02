# Project TODO

> **Navigation:** [Main README](../README.md) | [Pipeline Guide](RANKING_PIPELINE_GUIDE.md) | [Features](FEATURES.md) | [Clustering](CLUSTERING.md) | [Data Leakage](DATA_LEAKAGE.md) | [Testing](TESTING.md) | [Annual Statistics](ANNUAL_STATISTICS.md) | [Java Architecture](../java/docs/ARCHITECTURE.md) | [Java Data Sources](../java/docs/DATA_SOURCES.md)

---

## Active Items (roughly ordered by priority)

-   Java : Write NZX website scraper <br>
    This would include pdf parsing and feature extraction using natural language processing or a small large language model. 

-   Java : Investigate and cross reference data sources. <br>
    Leakage may come from the data itself, for example inaccurate timestamps. Cross reference against other data sources to ensure data aligns for a given timestamp. Should produce comprehensive document to prove this. 

-   Python : Improve pipeline performance. (Low prio) <br>     
    Pipeline is currently IO / single threaded slow operation limited. Improve this such that we spend as much time with 100% CPU / GPU useage.

-   Python : Fine tune model params. <br>
    Perform grid search with high granularity to find optimal params. 





