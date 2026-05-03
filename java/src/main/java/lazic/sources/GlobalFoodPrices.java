package lazic.sources;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.YearMonth;
import java.time.format.DateTimeFormatter;
import java.util.HashSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import lazic.utils.ingest.Cadence;
import lazic.utils.ingest.DataPoint;
import lazic.utils.ingest.DataSourceBase;
import lazic.utils.ingest.ReleaseDate;
import lazic.utils.ingest.ReleaseLag;
import lazic.utils.ingest.WebHtmlGetter;

/**
 * FAO Food Price Index data source.
 * 
 * Fetches global food price indices from FAO (Food and Agriculture Organization).
 * Provides monthly indices for: Food Price Index, Meat, Dairy, Cereals, Oils, Sugar.
 * Base period: 2014-2016=100
 * 
 * The URL requires the publication month, which changes monthly. This implementation
 * tries the current month first, then falls back to the previous month.
 * If all remote fetches fail, falls back to a local cache file.
 */
public class GlobalFoodPrices extends DataSourceBase {
	
	private static final String URL_TEMPLATE = 
		"https://www.fao.org/media/docs/worldfoodsituationlibraries/default-document-library/food_price_indices_data_csv_%s.csv";
	private static final String CURRENT_CSV_URL =
		"https://www.fao.org/media/docs/worldfoodsituationlibraries/default-document-library/food_price_indices_data_csv.csv?download=true";
	
	// Local fallback file path (relative to project root)
	private static final String LOCAL_FALLBACK_PATH = "data/fao_food_prices.csv";

	// FAO Food Price Index is published monthly, typically ~5 days after month-end.
	// Use a conservative 30-day lag to avoid look-ahead even when the FAO release schedule slips.
	// https://www.fao.org/worldfoodsituation/foodpricesindex/en/
	private static final ReleaseLag RELEASE_LAG = ReleaseLag.of(30);
	
	// Feature names for the 6 main price indices (columns 1-6 after Date)
	private static final String[] FEATURE_NAMES = {
		"FAO_FoodPriceIndex",
		"FAO_MeatPriceIndex",
		"FAO_DairyPriceIndex",
		"FAO_CerealsPriceIndex",
		"FAO_OilsPriceIndex",
		"FAO_SugarPriceIndex"
	};

	/**
	 * Returns a set of DataPoint's for FAO food price indices.
	 * Ticker is null as this is macroeconomic data.
	 */
	@Override
	public String getSourceName() { return "global_food_prices"; }

	@Override
	public Set<DataPoint> getDataPoints() {
		String rawData = fetchDataWithFallback();
		
		if (rawData == null || rawData.isEmpty()) {
			System.err.println("GlobalFoodPrices: Failed to fetch data from FAO");
			return new HashSet<>();
		}
		
		return parseCsvData(rawData);
	}
	
	/**
	 * Attempts to fetch data using current month, then previous month if that fails.
	 * Falls back to local file if all remote fetches fail.
	 */
	private String fetchDataWithFallback() {
		String result;
		for (String url : candidateUrls(YearMonth.now())) {
			result = tryFetchUrl(url);
			if (result != null) return result;
		}
		
		System.err.println("GlobalFoodPrices: All remote fetch attempts failed, trying local fallback...");
		
		// Try local file fallback
		result = tryLoadLocalFile();
		if (result != null) return result;
		
		System.err.println("GlobalFoodPrices: All fetch attempts failed (including local fallback)");
		return null;
	}
	
	/**
	 * Attempts to load data from a local CSV file.
	 */
	private String tryLoadLocalFile() {
		Path localPath = Paths.get(LOCAL_FALLBACK_PATH);
		System.out.println("GlobalFoodPrices: Attempting to load from local file: " + localPath.toAbsolutePath());
		
		if (!Files.exists(localPath)) {
			System.err.println("GlobalFoodPrices: Local file not found: " + localPath.toAbsolutePath());
			return null;
		}
		
		try {
			String data = Files.readString(localPath);
			System.out.println("GlobalFoodPrices: Loaded " + data.length() + " bytes from local file");
			
			if (isValidCsvData(data)) {
				System.out.println("GlobalFoodPrices: Valid CSV data found in local file");
				return data;
			} else {
				System.err.println("GlobalFoodPrices: Local file is not valid CSV");
				return null;
			}
		} catch (IOException e) {
			System.err.println("GlobalFoodPrices: Error reading local file: " + e.getMessage());
			return null;
		}
	}
	
	/**
	 * Attempts to fetch data from a specific month URL with detailed logging.
	 */
	static List<String> candidateUrls(YearMonth currentMonth) {
		List<String> urls = new ArrayList<>();
		urls.add(CURRENT_CSV_URL);
		urls.add(String.format(URL_TEMPLATE, formatMonthForUrl(currentMonth)));
		urls.add(String.format(URL_TEMPLATE, formatMonthForUrl(currentMonth.minusMonths(1))));
		urls.add(String.format(URL_TEMPLATE, formatMonthForUrl(currentMonth.minusMonths(2))));
		return urls;
	}

	private String tryFetchUrl(String url) {
		System.out.println("GlobalFoodPrices: Attempting to fetch from: " + url);
		
		try {
			String data = WebHtmlGetter.get(url);
			
			// Log response info
			if (data == null) {
				System.err.println("GlobalFoodPrices: Response was null for " + url);
				return null;
			}
			
			System.out.println("GlobalFoodPrices: Received " + data.length() + " bytes for " + url);
			
			// Show first 200 chars of response for debugging
			String preview = data.substring(0, Math.min(200, data.length())).replace("\n", "\\n").replace("\r", "\\r");
			System.out.println("GlobalFoodPrices: Response preview: [" + preview + "]");
			
			if (isValidCsvData(data)) {
				System.out.println("GlobalFoodPrices: Valid CSV data found for " + url);
				return data;
			} else {
				System.err.println("GlobalFoodPrices: Response for " + url + " is not valid CSV (missing 'FAO Food Price Index' header)");
				return null;
			}
		} catch (Exception e) {
			System.err.println("GlobalFoodPrices: Exception fetching " + url + ": " + e.getClass().getSimpleName() + ": " + e.getMessage());
			return null;
		}
	}
	
	/**
	 * Formats YearMonth to the URL format (e.g., "jan" for January).
	 */
	private static String formatMonthForUrl(YearMonth yearMonth) {
		// Format: 3-letter month abbreviation lowercase (e.g., "jan", "dec")
		return yearMonth.getMonth().toString().substring(0, 3).toLowerCase(Locale.ENGLISH);
	}
	
	/**
	 * Checks if the response looks like valid CSV data (not an error page).
	 */
	private boolean isValidCsvData(String data) {
		if (data == null || data.isEmpty()) {
			return false;
		}
		// Valid data should contain the header identifier
		return data.contains("FAO Food Price Index") || data.contains("Food Price Index");
	}
	
	private Set<DataPoint> parseCsvData(String rawData) {
		Set<DataPoint> dataPoints = new HashSet<>();
		
		String[] lines = rawData.split("\n");
		
		// Ticker is null for macroeconomic data
		final String ticker = null;
		
		// Debug counters
		int totalLines = lines.length;
		int skippedEmpty = 0;
		int skippedHeader = 0;
		int skippedTooFewColumns = 0;
		int skippedInvalidDate = 0;
		int parsedRows = 0;
		int parseErrors = 0;
		
		System.out.println("GlobalFoodPrices: Starting to parse " + totalLines + " lines");
		
		// Show first few lines for debugging
		System.out.println("GlobalFoodPrices: First 5 lines of data:");
		for (int i = 0; i < Math.min(5, lines.length); i++) {
			System.out.println("  Line " + i + ": [" + lines[i].substring(0, Math.min(80, lines[i].length())) + "...]");
		}
		
		for (int lineNum = 0; lineNum < lines.length; lineNum++) {
			String line = stripBom(lines[lineNum].trim());
			
			// Skip empty lines
			if (line.isEmpty() || line.replace(",", "").isBlank()) {
				skippedEmpty++;
				continue;
			}
			
			// Skip header lines
			if (line.contains("FAO Food Price Index") || line.startsWith("2014-2016") || line.startsWith("Date,")) {
				skippedHeader++;
				continue;
			}
			
			// Parse data rows (format: YYYY-MM,val1,val2,val3,val4,val5,val6,...)
			String[] parts = line.split(",", -1); // -1 to keep trailing empty strings
			
			if (parts.length < 7) {
				skippedTooFewColumns++;
				if (skippedTooFewColumns <= 3) {
					System.err.println("GlobalFoodPrices: Line " + lineNum + " has only " + parts.length + " columns (need 7): [" + line.substring(0, Math.min(50, line.length())) + "...]");
				}
				continue;
			}
			
			String dateStr = parts[0].trim();
			
			// Validate date format (YYYY-MM)
			if (!dateStr.matches("\\d{4}-\\d{2}")) {
				skippedInvalidDate++;
				if (skippedInvalidDate <= 3) {
					System.err.println("GlobalFoodPrices: Line " + lineNum + " has invalid date format: [" + dateStr + "] (expected YYYY-MM)");
				}
				continue;
			}
			
			try {
				LocalDateTime dateTime = parseMonthToDateTime(dateStr);
				parsedRows++;
				
				// Parse each of the 6 price indices
				for (int i = 0; i < FEATURE_NAMES.length; i++) {
					String valueStr = parts[i + 1].trim();
					
					if (valueStr.isEmpty()) continue;
					
					try {
						double value = Double.parseDouble(valueStr);
						dataPoints.add(new DataPoint(dateTime, ticker, FEATURE_NAMES[i], value));
					} catch (NumberFormatException e) {
						if (parseErrors < 5) {
							System.err.println("GlobalFoodPrices: Line " + lineNum + ", column " + (i+1) + " (" + FEATURE_NAMES[i] + "): Cannot parse value [" + valueStr + "]");
						}
						parseErrors++;
					}
				}
				
			} catch (Exception e) {
				parseErrors++;
				System.err.println("GlobalFoodPrices: Error parsing line " + lineNum + ": [" + line.substring(0, Math.min(60, line.length())) + "...] - " + e.getClass().getSimpleName() + ": " + e.getMessage());
			}
		}
		
		// Summary
		System.out.println("GlobalFoodPrices: Parse summary:");
		System.out.println("  Total lines:           " + totalLines);
		System.out.println("  Skipped (empty):       " + skippedEmpty);
		System.out.println("  Skipped (header):      " + skippedHeader);
		System.out.println("  Skipped (few columns): " + skippedTooFewColumns);
		System.out.println("  Skipped (bad date):    " + skippedInvalidDate);
		System.out.println("  Successfully parsed:   " + parsedRows + " rows");
		System.out.println("  Parse errors:          " + parseErrors);
		System.out.println("  Total data points:     " + dataPoints.size());
		
		if (dataPoints.isEmpty()) {
			System.err.println("GlobalFoodPrices: WARNING - No data points parsed! Check data format.");
		}
		
		return dataPoints;
	}
	
	/**
	 * Converts a monthly string (YYYY-MM) to LocalDateTime at the first day of that month.
	 */
	private LocalDateTime parseMonthToDateTime(String monthStr) {
		DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM");
		YearMonth yearMonth = YearMonth.parse(monthStr, formatter);
		LocalDateTime periodStart = yearMonth.atDay(1).atStartOfDay();
		return ReleaseDate.applyLag(periodStart, Cadence.MONTHLY, RELEASE_LAG);
	}

	private String stripBom(String line) {
		return line.startsWith("\uFEFF") ? line.substring(1) : line;
	}
}
