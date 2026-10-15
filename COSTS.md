# Costs and dependencies

Everything here is free for a public repo with no credit card. Terms below were not verified by the builder because the build environment cannot reach these sites. The sponsor verifies each source terms personally.

Terms verified on: (blank, sponsor to fill in per source)

| Dependency | Type | License or terms | Free limits | What breaks if terms change |
|---|---|---|---|---|
| pandas, numpy, pyarrow, pandera, pydantic, typer, structlog, PyYAML, requests | Python libraries | Open source (BSD, MIT, Apache) | None | Nothing, pin versions |
| pytest, hypothesis, ruff, black, mypy | Dev tools | Open source | None | Nothing |
| yfinance | Unofficial scraper of Yahoo | Library is open source. Yahoo data is for personal use. | Rate limited, unannounced breakage | Real price rebuild fails. CI unaffected. Yahoo derived data is never published. |
| FRED and ALFRED | API | Free key. Some series carry third party copyright. | Request rate limits | Macro series fail to rebuild. Use public domain series only. |
| SEC EDGAR | API | Public domain. Fair access policy. | 10 requests per second, descriptive User-Agent required | Fundamentals fail to rebuild |
| Ken French data library | Direct download | Free with citation | None stated | Factor rebuild fails. File layout can change. |
| AQR data library | Direct download | Free for research, check terms | None stated | Factor rebuild fails. File layout changes often. |
| CBOE delayed quotes | Undocumented JSON endpoints | Redistribution restricted | No history | Forward collection stops. Raw snapshots are never committed. |
| GDELT | Bulk files | Open | Large files | Tone signal unavailable. It is short sample by design. |
| Google Trends via pytrends | Unofficial | Grey | Heavy rate limiting | Attention signal falls back to abnormal volume only |
| Wikipedia S and P 500 changes | Public page | CC BY SA, attribution required | None | Universe reconstruction fails. Cached copy of the change log is the fallback. |
| GitHub Actions | CI | Free on public repos | Minutes quota | CI slows or stops. Local make all still works. |
| Streamlit Community Cloud | Dashboard hosting | Free tier | Resource limits | Dashboard runs locally instead |
