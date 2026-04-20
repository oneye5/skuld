package lazic.sources;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.google.gson.Gson;

import lazic.sources.config.Tickers;
import lazic.utils.ingest.DataPoint;
import lazic.utils.ingest.DataSourceBase;
import lazic.utils.ingest.WebHtmlGetter;

public class YfPrices extends DataSourceBase {

	private final String URL_TEMPLATE = "https://query1.finance.yahoo.com/v8/finance/chart/{TICKER}?interval=1d&period1=0&period2=99999999999&includeAdjustedClose=true&events=div,split";

	/**
	 * Returns a set of DataPoint's.
	 */
	@Override
	public String getSourceName() { return "yf_prices"; }

	@Override
	public Set<DataPoint> getDataPoints() {
		Set<DataPoint> dataPoints = new HashSet<>();
		String[] tickers = Tickers.TICKERS;
		Gson gson = new Gson();

		for (String ticker : tickers) {
			try {
				// 1. Construct URL
				String targetUrl = URL_TEMPLATE.replace("{TICKER}", ticker);

				// 2. Fetch Data
				System.out.println("Fetching data for: " + ticker);
				String rawData = WebHtmlGetter.get(targetUrl);

				if (rawData == null || rawData.isEmpty()) {
					System.err.println("No data received for " + ticker);
					continue;
				}

				// 3. Parse JSON using Inner DTOs
				YfResponse response = gson.fromJson(rawData, YfResponse.class);

				// 4. Validate response structure
				if (response.chart == null || response.chart.result == null || response.chart.result.isEmpty()) {
					System.err.println("Invalid JSON structure for " + ticker);
					continue;
				}

				Result result = response.chart.result.get(0);
				List<Long> timestamps = result.timestamp;
				Indicators indicators = result.indicators;

				// Validate data arrays exist
				if (timestamps == null || indicators == null || indicators.quote == null || indicators.quote.isEmpty()) {
					continue;
				}

				Quote quote = indicators.quote.get(0);

				// Get adjusted close if available
				List<Double> adjCloseValues = null;
				if (indicators.adjclose != null && !indicators.adjclose.isEmpty()) {
					adjCloseValues = indicators.adjclose.get(0).adjclose;
				}

				// 5. Iterate through time series and create DataPoints
				for (int i = 0; i < timestamps.size(); i++) {
					Long ts = timestamps.get(i);

					// Yahoo timestamps are in seconds, convert to LocalDateTime
					// Using system default zone, but you might prefer ZoneId.of("NZ") based on your data
					LocalDateTime date = LocalDateTime.ofInstant(Instant.ofEpochSecond(ts), ZoneId.systemDefault());

					// Extract features (handling potential nulls in the stream)
					addPoint(dataPoints, date, ticker, "Close", quote.close, i);
					addPoint(dataPoints, date, ticker, "Open", quote.open, i);
					addPoint(dataPoints, date, ticker, "High", quote.high, i);
					addPoint(dataPoints, date, ticker, "Low", quote.low, i);
					addPoint(dataPoints, date, ticker, "Volume", quote.volume, i);
					addPoint(dataPoints, date, ticker, "AdjClose", adjCloseValues, i);
				}

				// 6. Extract dividend events
				if (result.events != null && result.events.dividends != null) {
					for (DividendEvent div : result.events.dividends.values()) {
						LocalDateTime date = LocalDateTime.ofInstant(Instant.ofEpochSecond(div.date), ZoneId.systemDefault());
						dataPoints.add(new DataPoint(date, ticker, "Dividend", div.amount));
					}
				}

				// 7. Extract stock split events
				if (result.events != null && result.events.splits != null) {
					for (SplitEvent split : result.events.splits.values()) {
						LocalDateTime date = LocalDateTime.ofInstant(Instant.ofEpochSecond(split.date), ZoneId.systemDefault());
						// Store split ratio as numerator/denominator (e.g., 4:1 split = 4.0)
						double splitRatio = (double) split.numerator / split.denominator;
						dataPoints.add(new DataPoint(date, ticker, "Split", splitRatio));
					}
				}

			} catch (Exception e) {
				System.err.println("Error processing ticker " + ticker + ": " + e.getMessage());
				e.printStackTrace();
			}
		}

		return dataPoints;
	}

	// Helper to safely add a datapoint if the value exists at that index
	private void addPoint(Set<DataPoint> points, LocalDateTime date, String ticker, String feature, List<Double> values, int index) {
		if (values != null && index < values.size() && values.get(index) != null) {
			points.add(new DataPoint(date, ticker, feature, values.get(index)));
		}
	}

	// ============================================================
	// Internal DTO classes to map Yahoo Finance JSON structure
	// Structure: chart -> result[] -> [timestamp[], indicators -> quote[]]
	// ============================================================

	private static class YfResponse {
		Chart chart;
	}

	private static class Chart {
		List<Result> result;
		Object error; // can be mapped if needed
	}

	private static class Result {
		Meta meta;
		List<Long> timestamp;
		Indicators indicators;
		Events events;
	}

	private static class Meta {
		String currency;
		String symbol;
		String timezone;
		String exchangeName;
		String instrumentType;
		Double regularMarketPrice;
		Double previousClose;
		Double fiftyTwoWeekHigh;
		Double fiftyTwoWeekLow;
	}

	private static class Events {
		java.util.Map<String, DividendEvent> dividends;
		java.util.Map<String, SplitEvent> splits;
	}

	private static class DividendEvent {
		long date;
		double amount;
	}

	private static class SplitEvent {
		long date;
		double numerator;
		double denominator;
	}

	private static class Indicators {
		List<Quote> quote;
		List<AdjClose> adjclose;
	}

	private static class Quote {
		List<Double> open;
		List<Double> high;
		List<Double> low;
		List<Double> close;
		List<Double> volume;
	}

	private static class AdjClose {
		List<Double> adjclose;
	}
}