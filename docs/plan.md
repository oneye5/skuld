# Skuld
# High level purpose
The goal of this application is to provide statistically / mathmatically backed information to supplement my personal investing decisions.
# Means
The means of which this goal is acheived it to combine a few critical parts.
One part will need to be able to predict future outcomes of individual tickers in order to inform buy / sell / hold signals.
Another part will be portfolio management. Sharesies is the platform being used, so this project is subject to the fee structure that sharesies employs as well as limited to the individual tickers that sharesies allows for the buying and selling of.

A simplification of sharesies fee structure goes as follows:
5000$ worth of volume per month costing 15 dolar (highest teir monthly plan).

This imposes some important considerations, restructuring is ideally done once per month, and is capped at 5000 dolars worth of volume. Any volume after 5000 dolars is subject to full fee's at 1.9% which is significant and thus should be avoided where possible. However there may be cases, such as company colapse, that paying this fee may be worth it to avoid greater losses. 

The high level means of acheiving this outcome is to have a machine learning algorithm interpret a large set of data in order rank, clasify or regress on outcomes in order to inform what investing decisions are made. For example at the start of a new month, the inputs the application is given are: large amounts of data from the java portion of this application, the web scraper, my portfolio and from this the output should be actions to make, ie sell x shares of y, buy z shares of w.

# Methodology
Clean separation of conserns, clean use of contracts, particularly for transformation functions and directory structure. 

One 'application' per core process, for example the data fetcher is its own application in its own directory.

Leakage and integrity are core concerns. Measures should be put in place to avoid leakge of any kind. One such method could be in the transformer contract making it impossible to look forward, only being possible to look backward in time relative to the data point being transformed.

Performance, due to the amount of data, optomizing for low memory use and processing time is very important. Faster code = faster iteration speed = higher quality end product.

# Intended use case
Once a month I will run this application (including refreshing the input dataset via the java application) in order to rebalance my portolio, and use additional funds that I have deposited into my sharesies wallet. I will then read the output of this application, perform my own manual review double checking the recomendations of this application, and then executing on the recomendations asuming that they pass my own quality assurance criteria. By doing this I hope to make good use of the capital available to me, ideally beating all term deposits, high interest savings, bonds ect in average performance. 

# Risk
I am risk cautious, so thorough validation and backtesting are core to this project.

# Development mindset
Do not reinvent the wheel where it does not need to be reinvented. For example, I would prefer to use an existing backtesting library over creating our own, things of this nature. Use existing libraries where they exist and where it makes sense to do so. If the behaviour required is novel and libraries are not well suited then only then is it suitable to create our own systems.

Same goes for methodologies, the application should be research lead rather than based on arbitrary decisions. Decisions should be backed up by. 

# Tooling
Python dependency management and script execution via **`uv`** (Astral). All Python packages are managed as a `uv` workspace with a root lockfile. Scripts are run with `uv run`.
