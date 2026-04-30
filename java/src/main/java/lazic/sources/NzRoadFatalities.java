package lazic.sources;

import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.YearMonth;
import java.time.format.DateTimeFormatter;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import lazic.utils.ingest.Cadence;
import lazic.utils.ingest.DataPoint;
import lazic.utils.ingest.DataSourceBase;
import lazic.utils.ingest.ReleaseDate;
import lazic.utils.ingest.ReleaseLag;
import lazic.utils.ingest.WebHtmlGetter;

public class NzRoadFatalities extends DataSourceBase {
	private final String URL = "https://sdmx.oecd.org/public/rest/data/OECD.ITF,DSD_ST@DF_STFAT,1.0/NZL.M...ROAD...?dimensionAtObservation=AllDimensions";

	// NZ Ministry of Transport / OECD-ITF monthly road fatality stats: ~30 days after month-end.
	private static final ReleaseLag RELEASE_LAG = ReleaseLag.of(30);

	/**
	 * Returns a set of DataPoint's. Ticker is null if the datapoint does not pertain to a particular ticker,
	 * such as macroeconomic data for example. There are multiple DataPoint's in a time-series feature,
	 * and there may be multiple features returned overall.
	 */
	@Override
	public String getSourceName() { return "nz_road_fatalities"; }

	@Override
	public Set<DataPoint> getDataPoints() {
		String[] tickers = lazic.sources.config.Tickers.TICKERS;
		Gson gson = new Gson();
		String rawData = WebHtmlGetter.get(URL);

		try (FileWriter writer = new FileWriter("sample_data.txt")) {
			writer.write(rawData);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

		return parseRoadFatalitiesData(rawData, gson);
	}

	private Set<DataPoint> parseRoadFatalitiesData(String rawData, Gson gson) {
		Set<DataPoint> dataPoints = new HashSet<>();

		try {
			JsonObject root = gson.fromJson(rawData, JsonObject.class);
			JsonArray dataSets = root.getAsJsonArray("dataSets");

			if (dataSets == null || dataSets.size() == 0) {
				return dataPoints;
			}

			JsonObject dataSet = dataSets.get(0).getAsJsonObject();
			JsonObject observations = dataSet.getAsJsonObject("observations");

			// Get dimension metadata from structure
			JsonObject structure = root.getAsJsonObject("structure");
			JsonObject dimensions = structure.getAsJsonObject("dimensions");
			JsonArray observationDims = dimensions.getAsJsonArray("observation");

			// Build lookup map for time period dimension (index 8)
			Map<Integer, JsonObject> timePeriodMap = new java.util.HashMap<>();
			for (int i = 0; i < observationDims.size(); i++) {
				JsonObject dim = observationDims.get(i).getAsJsonObject();
				String dimId = dim.get("id").getAsString();

				if ("TIME_PERIOD".equals(dimId)) {
					JsonArray values = dim.getAsJsonArray("values");
					for (int j = 0; j < values.size(); j++) {
						timePeriodMap.put(j, values.get(j).getAsJsonObject());
					}
					break;
				}
			}

			// Parse each observation
			// Key format: "REF_AREA:FREQ:MEASURE:UNIT_MEASURE:TRANSPORT_MODE:GEO_COVERAGE:VEHICLE_TYPE:FUEL:TIME_PERIOD"
			// Example: "0:0:0:0:0:0:0:0:0" for NZL.M.FATALITIES.PS.ROAD._T._T._T.2025-04
			for (Map.Entry<String, JsonElement> entry : observations.entrySet()) {
				String key = entry.getKey();
				JsonArray obsValue = entry.getValue().getAsJsonArray();

				String[] indices = key.split(":");

				if (indices.length < 9) continue;

				// Extract time period index (position 8)
				int timePeriodIdx = Integer.parseInt(indices[8]);

				// Get time period (format: "2025-04" for monthly data)
				JsonObject timePeriodObj = timePeriodMap.get(timePeriodIdx);
				if (timePeriodObj == null) continue;

				String timePeriodStr = timePeriodObj.get("id").getAsString();

				// Get observation value (first element in array)
				if (obsValue.size() == 0 || obsValue.get(0).isJsonNull()) continue;
				double value = obsValue.get(0).getAsDouble();

				// Parse the time period string (format: "YYYY-MM")
				LocalDateTime timestamp = parseTimePeriod(timePeriodStr);
				if (timestamp == null) continue;

				// Feature name for NZ road fatalities
				String featureName = "nz_road_fatalities_monthly";

				// Ticker is null for macroeconomic/country-level data
				DataPoint dp = new DataPoint(timestamp, null, featureName, value);
				dataPoints.add(dp);
			}

		} catch (Exception e) {
			System.err.println("Error parsing road fatalities data: " + e.getMessage());
			e.printStackTrace();
		}

		return dataPoints;
	}

	/**
	 * Parse time period string in format "YYYY-MM" to LocalDateTime
	 * Sets the day to the first of the month and time to midnight
	 */
	private LocalDateTime parseTimePeriod(String timePeriodStr) {
		try {
			// Handle format like "2025-04"
			YearMonth yearMonth = YearMonth.parse(timePeriodStr, DateTimeFormatter.ofPattern("yyyy-MM"));
			LocalDateTime periodStart = yearMonth.atDay(1).atStartOfDay();
			return ReleaseDate.applyLag(periodStart, Cadence.MONTHLY, RELEASE_LAG);
		} catch (Exception e) {
			System.err.println("Failed to parse time period: " + timePeriodStr);
			return null;
		}
	}
}