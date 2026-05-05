package lazic.sources;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import lazic.sources.config.Tickers;
import lazic.utils.ingest.DataPoint;
import lazic.utils.ingest.DataSourceBase;
import lazic.utils.ingest.WebHtmlGetter;

/**
 * Fetches GICS sector classification for each ticker from the Yahoo Finance
 * finance/search endpoint. Sector is stored as the GICS numeric code (Double):
 * 10=Energy, 15=Materials, 20=Industrials, 25=ConsumerDiscretionary,
 * 30=ConsumerStaples, 35=HealthCare, 40=Financials, 45=IT,
 * 50=CommunicationServices, 55=Utilities, 60=RealEstate.
 *
 * Timestamp is set to 2010-01-01 so the sector classification is PIT-available
 * from the start of the backtest window.
 */
public class YfSector extends DataSourceBase {

    private static final LocalDateTime SECTOR_TIMESTAMP =
        LocalDate.of(2010, 1, 1).atStartOfDay();

    /**
     * Maps Yahoo Finance sector name strings to GICS sector numeric codes.
     * Yahoo Finance uses GICS-based names but with slightly different labels.
     */
    private static final Map<String, Double> SECTOR_CODES = Map.ofEntries(
        Map.entry("Energy",                 10.0),
        Map.entry("Basic Materials",        15.0),
        Map.entry("Industrials",            20.0),
        Map.entry("Consumer Cyclical",      25.0),
        Map.entry("Consumer Defensive",     30.0),
        Map.entry("Healthcare",             35.0),
        Map.entry("Health Care",            35.0),
        Map.entry("Financial Services",     40.0),
        Map.entry("Technology",             45.0),
        Map.entry("Communication Services", 50.0),
        Map.entry("Utilities",              55.0),
        Map.entry("Real Estate",            60.0)
    );

    private final Function<String, String> fetcher;

    /** Production constructor. */
    public YfSector() {
        this(WebHtmlGetter::get);
    }

    /** Package-private constructor for tests. */
    YfSector(Function<String, String> fetcher) {
        super();
        this.fetcher = fetcher;
    }

    @Override
    public String getSourceName() { return "yf_sector"; }

    @Override
    public Set<DataPoint> getDataPoints() {
        Set<DataPoint> points = new HashSet<>();
        for (String ticker : Tickers.TICKERS) {
            try {
                String json = fetcher.apply(buildUrl(ticker));
                points.addAll(parseSector(json, ticker));
            } catch (Exception e) {
                System.err.println("YfSector: error for " + ticker + " — " + e.getMessage());
            }
        }
        return points;
    }

    // ── Package-private for tests ────────────────────────────────────────

    static String buildUrl(String ticker) {
        return BASE_URL.replace("{TICKER}", ticker);
    }

    /**
     * Parses a Yahoo Finance finance/search JSON response.
     * Returns an empty set (never throws) on any malformed or null input.
     * Emits a DataPoint(SECTOR_TIMESTAMP, ticker, "gics_sector", gicsCode)
     * for each ticker whose sector maps to a known GICS code.
     */
    static Set<DataPoint> parseSector(String json, String ticker) {
        Set<DataPoint> points = new HashSet<>();
        if (json == null || json.isBlank()) return points;

        JsonObject root;
        try {
            root = new Gson().fromJson(json, JsonObject.class);
        } catch (Exception e) {
            System.err.println("YfSector: failed to parse JSON for " + ticker + " — " + e.getMessage());
            return points;
        }

        if (!root.has("quotes") || root.get("quotes").isJsonNull()) return points;
        JsonArray quotes = root.getAsJsonArray("quotes");

        JsonObject exactMatch = null;
        for (int i = 0; i < quotes.size(); i++) {
            if (quotes.get(i) == null || quotes.get(i).isJsonNull()) continue;

            JsonObject quote = quotes.get(i).getAsJsonObject();
            if (!quote.has("symbol") || quote.get("symbol").isJsonNull()) continue;

            if (ticker.equalsIgnoreCase(quote.get("symbol").getAsString().strip())) {
                exactMatch = quote;
                break;
            }
        }
        if (exactMatch == null) return points;

        if (!exactMatch.has("sector") || exactMatch.get("sector").isJsonNull()) return points;
        String sectorStr = exactMatch.get("sector").getAsString().strip();
        if (sectorStr.isEmpty()) return points;

        Double gicsCode = SECTOR_CODES.get(sectorStr);
        if (gicsCode == null) {
            System.err.println("YfSector: unmapped sector '" + sectorStr + "' for " + ticker);
            return points;
        }

        points.add(new DataPoint(SECTOR_TIMESTAMP, ticker, "gics_sector", gicsCode));
        return points;
    }

    // ── URL ─────────────────────────────────────────────────────────────

    private static final String BASE_URL =
        "https://query1.finance.yahoo.com/v1/finance/search?q={TICKER}";
}
