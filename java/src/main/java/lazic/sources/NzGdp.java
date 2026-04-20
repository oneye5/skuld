package lazic.sources;

import java.time.LocalDateTime;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import lazic.utils.ingest.DataPoint;
import lazic.utils.ingest.DataSourceBase;
import lazic.utils.ingest.WebHtmlGetter;

public class NzGdp extends DataSourceBase {
	final String URL = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_NATIO_CURR,1.1/Q..NZL.S13+S14.........?startPeriod=2000-Q1&dimensionAtObservation=AllDimensions&format=genericdata";

	/**
	 * Returns a set of DataPoint's. Ticker is null if the datapoint does not pertain to a particular ticker, such as macroeconomic data for example
	 * There are multiple DataPoint's in a time-series feature, and there may be multiple features returned overall.
	 */
	@Override
	public String getSourceName() { return "nz_gdp"; }

	@Override
	public Set<DataPoint> getDataPoints() {
		String rawData = WebHtmlGetter.get(URL);
		return parseXmlData(rawData);
	}

	/**
	 * Parses the SDMX-ML XML raw data.
	 * Extracts TIME_PERIOD, SECTOR, TRANSACTION, ObsValue, and UNIT_MULT.
	 */
	private Set<DataPoint> parseXmlData(String rawData) {
		Set<DataPoint> dataPoints = new HashSet<>();

		// Updated Regex to capture TIME_PERIOD, SECTOR, TRANSACTION, ObsValue, and UNIT_MULT.
		// Group 1: TIME_PERIOD value (e.g., "2023-Q2")
		// Group 2: SECTOR value (e.g., "S13" or "S14")
		// Group 3: TRANSACTION value (e.g., "P51G" or "P3")
		// Group 4: ObsValue (the data value, e.g., "6569")
		// Group 5: UNIT_MULT value (e.g., "6")
		String regex = "<generic:Obs>.*?<generic:Value id=\"TIME_PERIOD\" value=\"(.*?)\" />.*?" +
						"<generic:Value id=\"SECTOR\" value=\"(.*?)\" />.*?" +
						"<generic:Value id=\"TRANSACTION\" value=\"(.*?)\" />.*?" +
						"</generic:ObsKey><generic:ObsValue value=\"(.*?)\" />.*?<generic:Value id=\"UNIT_MULT\" value=\"(\\d)\" />";

		Pattern pattern = Pattern.compile(regex, Pattern.DOTALL);
		Matcher matcher = pattern.matcher(rawData);

		// Ticker is null for macroeconomic data
		final String ticker = null;

		// Look up map for better feature names (example, extend as needed)
		final java.util.Map<String, String> sectorMap = new java.util.HashMap<>();
		sectorMap.put("S13", "GeneralGovernment");
		sectorMap.put("S14", "NFISH"); // Non-profit institutions serving households

		final java.util.Map<String, String> transactionMap = new java.util.HashMap<>();
		transactionMap.put("P51G", "GrossFixedCapitalFormation");
		transactionMap.put("P3", "FinalConsumptionExpenditure");
		// Add other relevant transaction codes from the data as needed

		while (matcher.find()) {
			String timePeriodStr = matcher.group(1);
			String sectorCode = matcher.group(2);
			String transactionCode = matcher.group(3);
			String obsValueStr = matcher.group(4);
			String unitMultiplierStr = matcher.group(5);

			// Construct a meaningful feature name
			String sectorName = sectorMap.getOrDefault(sectorCode, sectorCode);
			String transactionName = transactionMap.getOrDefault(transactionCode, transactionCode);

			// Example: NZL_GeneralGovernment_GrossFixedCapitalFormation_MillionsNZD
			final String featureName = "NZL_" + sectorName + "_" + transactionName + "_MillionsNZD";

			try {
				// Convert the "YYYY-QX" quarter string to a LocalDateTime at the start of the quarter
				LocalDateTime dateTime = convertQuarterToDateTime(timePeriodStr);

				// Convert the value string to a Double, applying the unit multiplier (e.g., 10^6 for Millions)
				int unitMultiplierPower = Integer.parseInt(unitMultiplierStr);
				double rawValue = Double.parseDouble(obsValueStr);
				double finalValue = rawValue * Math.pow(10, unitMultiplierPower);

				dataPoints.add(new DataPoint(dateTime, ticker, featureName, finalValue));

			} catch (Exception e) {
				// Log or handle parsing errors if necessary
				System.err.println("Error parsing data point: " + timePeriodStr + ", " + obsValueStr +
								", Sector: " + sectorCode + ", Transaction: " + transactionCode + ": " + e.getMessage());
			}
		}

		return dataPoints;
	}

	/**
	 * Converts a quarterly string (YYYY-QX) to a LocalDateTime at the start of that quarter.
	 */
	private LocalDateTime convertQuarterToDateTime(String quarterStr) {
		String[] parts = quarterStr.split("-Q");
		int year = Integer.parseInt(parts[0]);
		int quarter = Integer.parseInt(parts[1]);

		int month = switch (quarter) {
			case 1 -> 1; // Q1 starts Jan 1st
			case 2 -> 4; // Q2 starts Apr 1st
			case 3 -> 7; // Q3 starts Jul 1st
			case 4 -> 10; // Q4 starts Oct 1st
			default -> throw new IllegalArgumentException("Invalid quarter: " + quarter);
		};

		// Assuming the data point is for the first day of the quarter
		return LocalDateTime.of(year, month, 1, 0, 0);
	}
}