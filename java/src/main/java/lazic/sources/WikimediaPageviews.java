package lazic.sources;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.gson.Gson;
import com.google.gson.annotations.SerializedName;

import lazic.utils.ingest.Cadence;
import lazic.utils.ingest.DataPoint;
import lazic.utils.ingest.DataSourceBase;
import lazic.utils.ingest.ReleaseDate;
import lazic.utils.ingest.ReleaseLag;
import lazic.utils.ingest.WebHtmlGetter;

/**
 * Fetches Wikipedia pageviews data for NZX stocks and macro indicators.
 * 
 * <h2>Data source</h2>
 * Wikimedia REST API - Historical data available from July 1, 2015.
 * 
 * <h2>Configuration</h2>
 * config/wikimedia_pages.csv - Mapping of tickers to Wikipedia pages
 * 
 * <h2>Output features</h2>
 * <ul>
 *   <li>Wiki_Views: Total pageviews (all-access)</li>
 *   <li>Wiki_Views_Desktop: Desktop browser views</li>
 *   <li>Wiki_Views_Mobile: Mobile views (mobile-web + mobile-app)</li>
 * </ul>
 * 
 * <h2>Research basis</h2>
 * Wikipedia pageviews have been shown to predict financial markets:
 * <ul>
 *   <li>Moat et al. (2013) - "Quantifying Wikipedia Usage Patterns Before Stock Market Moves"
 *       showed Wikipedia company page views predict stock price movements</li>
 *   <li>Preis et al. (2013) - "Quantifying Trading Behavior in Financial Markets Using Google Trends"
 *       showed searches for "debt" predict market declines</li>
 *   <li>Da et al. (2011) - Retail investor attention predicts short-term returns</li>
 * </ul>
 * 
 * <h2>Page categories in config</h2>
 * <ul>
 *   <li><b>Company pages</b>: Direct attention/sentiment for NZX stocks</li>
 *   <li><b>Fear indicators</b>: Crisis-related pages spike during market stress</li>
 *   <li><b>Commodities</b>: Relevant to NZ export economy (dairy, wool, forestry)</li>
 *   <li><b>Monetary policy</b>: Central bank and interest rate attention</li>
 *   <li><b>Consumer sentiment</b>: Retail and housing market proxies</li>
 *   <li><b>Global trade</b>: Trade war, tariff, and supply chain attention</li>
 *   <li><b>Technology</b>: Growth sector sentiment indicators</li>
 *   <li><b>Regional</b>: Australia and China (NZ's major trading partners)</li>
 * </ul>
 */
public class WikimediaPageviews extends DataSourceBase {

    private static final String CONFIG_PATH = "/lazic/sources/config/wikimedia_pages.csv";
    
    private static final String API_TEMPLATE = 
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/{ACCESS}/all-agents/{ARTICLE}/daily/{START}/{END}";
    
    // Data starts July 1, 2015
    private static final String START_DATE = "2015070100";
    private static final String END_DATE = "2099123100";
    
    private static final DateTimeFormatter TIMESTAMP_PARSER = 
        DateTimeFormatter.ofPattern("yyyyMMddHH");

    // Wikimedia daily pageview API publishes the previous day's totals roughly 1 day after period end.
    // https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews
    private static final ReleaseLag RELEASE_LAG = ReleaseLag.of(1);

    @Override
    public String getSourceName() { return "wikimedia_pageviews"; }

    @Override
    public Set<DataPoint> getDataPoints() {
        Set<DataPoint> dataPoints = new HashSet<>();
        Gson gson = new Gson();
        
        // Load page mappings from CSV
        List<PageMapping> mappings = loadConfig();
        
        if (mappings.isEmpty()) {
            System.err.println("WikimediaPageviews: No page mappings found in config");
            return dataPoints;
        }
        
        System.out.println("WikimediaPageviews: Loaded " + mappings.size() + " page mappings");
        
        int totalPages = mappings.size();
        int currentPage = 0;
        
        for (PageMapping mapping : mappings) {
            currentPage++;
            try {
                System.out.println("[" + currentPage + "/" + totalPages + "] Fetching Wikipedia pageviews for: " + mapping.wikipediaPage);
                
                // Fetch all-access (total)
                Map<LocalDateTime, Long> totalViews = fetchPageviews(
                    mapping.wikipediaPage, "all-access", gson
                );
                
                // Fetch desktop
                Map<LocalDateTime, Long> desktopViews = fetchPageviews(
                    mapping.wikipediaPage, "desktop", gson
                );
                
                // Fetch mobile-web
                Map<LocalDateTime, Long> mobileWebViews = fetchPageviews(
                    mapping.wikipediaPage, "mobile-web", gson
                );
                
                // Fetch mobile-app
                Map<LocalDateTime, Long> mobileAppViews = fetchPageviews(
                    mapping.wikipediaPage, "mobile-app", gson
                );
                
                // Create DataPoints
                // ticker is null for macro data (empty ticker in CSV)
                String ticker = (mapping.ticker == null || mapping.ticker.isEmpty()) 
                    ? null 
                    : mapping.ticker;
                
                // Feature name prefix for macro data
                String featurePrefix = (ticker == null) 
                    ? mapping.wikipediaPage + "_" 
                    : "";
                
                // Add total views
                for (Map.Entry<LocalDateTime, Long> entry : totalViews.entrySet()) {
                    dataPoints.add(new DataPoint(
                        entry.getKey(),
                        ticker,
                        featurePrefix + "Wiki_Views",
                        entry.getValue().doubleValue()
                    ));
                }
                
                // Add desktop views
                for (Map.Entry<LocalDateTime, Long> entry : desktopViews.entrySet()) {
                    dataPoints.add(new DataPoint(
                        entry.getKey(),
                        ticker,
                        featurePrefix + "Wiki_Views_Desktop",
                        entry.getValue().doubleValue()
                    ));
                }
                
                // Add mobile views (mobile-web + mobile-app combined)
                Set<LocalDateTime> allMobileDates = new HashSet<>();
                allMobileDates.addAll(mobileWebViews.keySet());
                allMobileDates.addAll(mobileAppViews.keySet());
                
                for (LocalDateTime date : allMobileDates) {
                    long mobileWeb = mobileWebViews.getOrDefault(date, 0L);
                    long mobileApp = mobileAppViews.getOrDefault(date, 0L);
                    long totalMobile = mobileWeb + mobileApp;
                    
                    dataPoints.add(new DataPoint(
                        date,
                        ticker,
                        featurePrefix + "Wiki_Views_Mobile",
                        (double) totalMobile
                    ));
                }
                
                System.out.println("  -> Added " + totalViews.size() + " days of data");
                
            } catch (Exception e) {
                System.err.println("Error fetching pageviews for " + mapping.wikipediaPage + ": " + e.getMessage());
            }
        }
        
        return dataPoints;
    }
    
    /**
     * Fetch pageviews for a specific article and access type.
     */
    private Map<LocalDateTime, Long> fetchPageviews(String article, String access, Gson gson) {
        Map<LocalDateTime, Long> result = new HashMap<>();
        
        String url = API_TEMPLATE
            .replace("{ACCESS}", access)
            .replace("{ARTICLE}", article)
            .replace("{START}", START_DATE)
            .replace("{END}", END_DATE);
        
        try {
            String rawData = WebHtmlGetter.get(url);
            
            if (rawData == null || rawData.isEmpty()) {
                return result;
            }
            
            PageviewResponse response = gson.fromJson(rawData, PageviewResponse.class);
            
            if (response.items == null) {
                return result;
            }
            
            for (PageviewItem item : response.items) {
                try {
                    // Parse timestamp (format: "2015070100"), then shift to release date.
                    LocalDateTime periodStart = LocalDateTime.parse(item.timestamp, TIMESTAMP_PARSER);
                    LocalDateTime date = ReleaseDate.applyLag(periodStart, Cadence.DAILY, RELEASE_LAG);
                    result.put(date, item.views);
                } catch (Exception e) {
                    // Skip malformed timestamps
                }
            }
            
        } catch (Exception e) {
            System.err.println("  Error fetching " + access + " views: " + e.getMessage());
        }
        
        return result;
    }
    
    /**
     * Load page mappings from CSV config file.
     */
    private List<PageMapping> loadConfig() {
        List<PageMapping> mappings = new ArrayList<>();
        
        try (InputStream is = getClass().getResourceAsStream(CONFIG_PATH)) {
            if (is == null) {
                System.err.println("WikimediaPageviews: Config file not found: " + CONFIG_PATH);
                return mappings;
            }
            
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(is, StandardCharsets.UTF_8))) {
                
                String line;
                boolean isHeader = true;
                
                while ((line = reader.readLine()) != null) {
                    // Skip header
                    if (isHeader) {
                        isHeader = false;
                        continue;
                    }
                    
                    // Skip empty lines and comments
                    if (line.trim().isEmpty() || line.trim().startsWith("#")) {
                        continue;
                    }
                    
                    // Parse CSV line: ticker,wikipedia_page
                    String[] parts = line.split(",", -1); // -1 to keep empty strings
                    
                    if (parts.length >= 2) {
                        String ticker = parts[0].trim();
                        String wikipediaPage = parts[1].trim();
                        
                        if (!wikipediaPage.isEmpty()) {
                            mappings.add(new PageMapping(ticker, wikipediaPage));
                        }
                    }
                }
            }
            
        } catch (Exception e) {
            System.err.println("Error loading config: " + e.getMessage());
        }
        
        return mappings;
    }
    
    // ========== Inner Classes ==========
    
    /**
     * Mapping between NZX ticker and Wikipedia page.
     */
    private static class PageMapping {
        final String ticker;        // May be null/empty for macro data
        final String wikipediaPage; // URL-encoded page title
        
        PageMapping(String ticker, String wikipediaPage) {
            this.ticker = ticker;
            this.wikipediaPage = wikipediaPage;
        }
    }
    
    // ========== JSON Response DTOs ==========
    
    private static class PageviewResponse {
        List<PageviewItem> items;
    }
    
    private static class PageviewItem {
        String project;
        String article;
        String granularity;
        String timestamp;
        String access;
        String agent;
        @SerializedName("views")
        Long views;
    }
}
