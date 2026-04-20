package lazic.sources;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.gson.Gson;

import lazic.utils.ingest.DataPoint;
import lazic.utils.ingest.DataSourceBase;
import lazic.utils.ingest.WebHtmlGetter;

public class NzRatesFx extends DataSourceBase {

	// Note: The URL fetches Financial Market data (Interest rates, Exchange rates).
	private final String URL = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/NZL.M..PA.....?dimensionAtObservation=AllDimensions&format=jsondata";

	@Override
	public String getSourceName() { return "nz_rates_fx"; }

	@Override
	public Set<DataPoint> getDataPoints() {
		Set<DataPoint> dataPoints = new HashSet<>();

		// 1. Fetch Raw Data
		String rawData = WebHtmlGetter.get(URL);

		if (rawData == null || rawData.isEmpty()) {
			System.err.println("No data retrieved from OECD source.");
			return dataPoints;
		}

		// 2. Parse JSON structure
		Gson gson = new Gson();
		SdmxResponse response = gson.fromJson(rawData, SdmxResponse.class);

		// Safety checks to prevent NullPointerExceptions
		if (response == null || response.data == null
						|| response.data.dataSets == null || response.data.dataSets.isEmpty()
						|| response.data.structures == null || response.data.structures.isEmpty()) {
			return dataPoints;
		}

		// 3. Extract Dimension lookups
		// The JSON contains a list of structures. We want the first one (index 0).
		Structure structure = response.data.structures.get(0);

		if (structure.dimensions == null || structure.dimensions.observation == null) {
			return dataPoints;
		}

		List<Dimension> dimensions = structure.dimensions.observation;

		// Based on the provided JSON source, the key positions in the observation string "0:0:0..." are:
		// Index 2 = MEASURE (e.g., "Long-term interest rates")
		// Index 9 = TIME_PERIOD (e.g., "2025-02")
		if (dimensions.size() <= 9) {
			System.err.println("Dimensions list is too short to parse.");
			return dataPoints;
		}

		Dimension measureDim = dimensions.get(2);
		Dimension timeDim = dimensions.get(9);

		// 4. Iterate through observations and map to DataPoints
		Map<String, List<Double>> observations = response.data.dataSets.get(0).observations;

		if (observations == null) return dataPoints;

		for (Map.Entry<String, List<Double>> entry : observations.entrySet()) {
			String key = entry.getKey(); // e.g. "0:0:2:0:0:0:0:0:0:12"
			List<Double> values = entry.getValue();

			// The value array's 0 index holds the actual data value
			if (values == null || values.isEmpty() || values.get(0) == null) continue;
			Double value = values.get(0);

			// Parse key indices
			String[] indices = key.split(":");

			try {
				int measureIndex = Integer.parseInt(indices[2]);
				int timeIndex = Integer.parseInt(indices[9]);

				// Lookup relevant metadata
				// Get feature name from dimensions (e.g., "Long-term interest rates")
				String featureName = measureDim.values.get(measureIndex).name;

				// Get time information
				DimensionValue timeVal = timeDim.values.get(timeIndex);
				String timeStartStr = timeVal.start;

				// Parse Time
				LocalDateTime timestamp;
				if (timeStartStr != null) {
					timestamp = LocalDateTime.parse(timeStartStr, DateTimeFormatter.ISO_DATE_TIME);
				} else {
					// Fallback: if 'start' is missing, construct from 'id' (e.g., "2025-02")
					String timeId = timeVal.id;
					// Append day and time for monthly data parsing
					timestamp = LocalDateTime.parse(timeId + "-01T00:00:00");
				}

				// Create DataPoint
				// Ticker is null as this is Macro data
				dataPoints.add(new DataPoint(
								timestamp,
								null,
								featureName,
								value
				));

			} catch (Exception e) {
				// Log parsing errors but continue processing other points
				// System.err.println("Skipping point " + key + ": " + e.getMessage());
			}
		}

		return dataPoints;
	}

	// ==========================================
	// Internal POJOs for GSON Parsing
	// ==========================================

	private static class SdmxResponse {
		public Data data;
	}

	private static class Data {
		public List<DataSet> dataSets;

		// Note: The JSON uses "structures" (plural/list), not "structure"
		public List<Structure> structures;
	}

	private static class DataSet {
		// Maps keys like "0:0:0..." to a list [Value, Status, etc.]
		public Map<String, List<Double>> observations;
	}

	private static class Structure {
		public Dimensions dimensions;
	}

	private static class Dimensions {
		// JSON path: data -> structures[0] -> dimensions -> observation
		public List<Dimension> observation;
	}

	private static class Dimension {
		public String id;
		public String name;
		public List<DimensionValue> values;
	}

	private static class DimensionValue {
		public String id;
		public String name;
		// "start" is specific to the TIME_PERIOD dimension in SDMX-JSON
		public String start;
	}
}