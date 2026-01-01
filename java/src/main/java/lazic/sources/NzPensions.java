package lazic.sources;

import com.google.gson.Gson;
import lazic.utils.ingest.DataPoint;
import lazic.utils.ingest.DataSourceBase;
import lazic.utils.ingest.WebHtmlGetter;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

public class NzPensions extends DataSourceBase {
	private final String URL = "https://sdmx.oecd.org/public/rest/data/OECD.DAF.CM,DSD_FP@DF_FPS,1.0/NZL.A.1121+1131+1141+1151+1210+1215+1230+1240+1245+1250+1255+1270+1000.._T._T._T?startPeriod=2001&dimensionAtObservation=AllDimensions";

	/**
	 * Returns a set of DataPoint's. Ticker is null if the datapoint does not pertain to a particular ticker, such as macroeconomic data for example
	 * There are multiple DataPoint's in a time-series feature, and there may be multiple features returned overall.
	 */
	@Override
	public Set<DataPoint> getDataPoints() {
		Set<DataPoint> dataPoints = new HashSet<>();

		// 1. Fetch Raw Data
		String rawData = WebHtmlGetter.get(URL);

		if (rawData == null || rawData.isEmpty()) {
			System.err.println("NzPensions: No data retrieved from OECD source.");
			return dataPoints;
		}

		// 2. Parse JSON structure
		Gson gson = new Gson();
		SdmxResponse response;

		try {
			response = gson.fromJson(rawData, SdmxResponse.class);
		} catch (Exception e) {
			System.err.println("NzPensions: Failed to parse JSON: " + e.getMessage());
			return dataPoints;
		}

		// Safety checks to prevent NullPointerExceptions
		if (response == null || response.dataSets == null || response.dataSets.isEmpty()
						|| response.structure == null || response.structure.dimensions == null
						|| response.structure.dimensions.observation == null) {
			System.err.println("NzPensions: Invalid data structure received.");
			return dataPoints;
		}

		// 3. Extract Dimension lookups
		List<Dimension> dimensions = response.structure.dimensions.observation;

		// Based on the SDMX structure, observation keys are formatted as "0:0:2:0:0:0:0:0"
		// Positions map to: REF_AREA(0):FREQ(1):MEASURE(2):UNIT_MEASURE(3):PLAN_TYPE(4):DEFINITION_TYPE(5):VEHICLE_TYPE(6):TIME_PERIOD(7)
		// We need MEASURE (position 2), UNIT_MEASURE (position 3), and TIME_PERIOD (position 7)
		if (dimensions.size() <= 7) {
			System.err.println("NzPensions: Dimensions list is too short to parse.");
			return dataPoints;
		}

		Dimension measureDim = dimensions.get(2);
		Dimension unitMeasureDim = dimensions.get(3);
		Dimension timeDim = dimensions.get(7);

		if (measureDim == null || measureDim.values == null
						|| unitMeasureDim == null || unitMeasureDim.values == null
						|| timeDim == null || timeDim.values == null) {
			System.err.println("NzPensions: Required dimensions are missing.");
			return dataPoints;
		}

		// 4. Iterate through observations and map to DataPoints
		Map<String, List<Double>> observations = response.dataSets.get(0).observations;

		if (observations == null || observations.isEmpty()) {
			System.err.println("NzPensions: No observations found in dataset.");
			return dataPoints;
		}

		int successCount = 0;
		int skippedCount = 0;

		for (Map.Entry<String, List<Double>> entry : observations.entrySet()) {
			String key = entry.getKey(); // e.g. "0:0:2:0:0:0:0:1"
			List<Double> values = entry.getValue();

			// The value array's 0 index holds the actual data value
			if (values == null || values.isEmpty() || values.get(0) == null) {
				skippedCount++;
				continue;
			}

			Double value = values.get(0);

			// Skip zero values as they typically indicate missing or not applicable data
			if (value == 0.0) {
				skippedCount++;
				continue;
			}

			// Parse key indices
			String[] indices = key.split(":");

			if (indices.length < 8) {
				skippedCount++;
				continue;
			}

			try {
				int measureIndex = Integer.parseInt(indices[2]);
				int unitMeasureIndex = Integer.parseInt(indices[3]);
				int timeIndex = Integer.parseInt(indices[7]);

				// Validate indices are within bounds
				if (measureIndex >= measureDim.values.size()
								|| unitMeasureIndex >= unitMeasureDim.values.size()
								|| timeIndex >= timeDim.values.size()) {
					skippedCount++;
					continue;
				}

				// Lookup relevant metadata
				String measureName = measureDim.values.get(measureIndex).name;
				String unitMeasureName = unitMeasureDim.values.get(unitMeasureIndex).name;

				// Construct feature name combining measure and unit
				String featureName = "NZ_Pensions_" + measureName + " (" + unitMeasureName + ")";

				// Get time information
				DimensionValue timeVal = timeDim.values.get(timeIndex);

				// Parse Time - handle both 'start' and 'id' formats
				LocalDateTime timestamp;
				if (timeVal.start != null && !timeVal.start.isEmpty()) {
					try {
						timestamp = LocalDateTime.parse(timeVal.start, DateTimeFormatter.ISO_DATE_TIME);
					} catch (Exception e) {
						// Fallback to constructing from id
						timestamp = parseTimeFromId(timeVal.id);
					}
				} else if (timeVal.id != null && !timeVal.id.isEmpty()) {
					timestamp = parseTimeFromId(timeVal.id);
				} else {
					skippedCount++;
					continue;
				}

				// Create DataPoint (ticker is null for macro data)
				dataPoints.add(new DataPoint(
								timestamp,
								null,
								featureName,
								value
				));
				successCount++;

			} catch (NumberFormatException e) {
				System.err.println("NzPensions: Failed to parse indices for key: " + key);
				skippedCount++;
			} catch (Exception e) {
				System.err.println("NzPensions: Error processing key " + key + ": " + e.getMessage());
				skippedCount++;
			}
		}

		System.out.println("NzPensions: Successfully parsed " + successCount + " data points, skipped " + skippedCount);
		return dataPoints;
	}

	/**
	 * Parse time from ID string (e.g., "2019", "2019-01")
	 * Annual data comes as "YYYY", so we default to January 1st
	 */
	private LocalDateTime parseTimeFromId(String timeId) {
		if (timeId == null || timeId.isEmpty()) {
			throw new IllegalArgumentException("Time ID is null or empty");
		}

		// Handle annual data (e.g., "2019")
		if (timeId.length() == 4) {
			return LocalDateTime.parse(timeId + "-01-01T00:00:00");
		}

		// Handle monthly data (e.g., "2019-01")
		if (timeId.length() == 7) {
			return LocalDateTime.parse(timeId + "-01T00:00:00");
		}

		// Handle full date (e.g., "2019-01-01")
		if (timeId.length() == 10) {
			return LocalDateTime.parse(timeId + "T00:00:00");
		}

		throw new IllegalArgumentException("Unsupported time ID format: " + timeId);
	}

	// ==========================================
	// Internal POJOs for GSON Parsing
	// ==========================================

	private static class SdmxResponse {
		public List<DataSet> dataSets;
		public Structure structure;
	}

	private static class DataSet {
		// Maps keys like "0:0:0:0:0:0:0:0" to a list [Value, Status, etc.]
		public Map<String, List<Double>> observations;
	}

	private static class Structure {
		public String name;
		public String description;
		public Dimensions dimensions;
	}

	private static class Dimensions {
		// JSON path: structure -> dimensions -> observation
		public List<Dimension> observation;
	}

	private static class Dimension {
		public String id;
		public String name;
		public Integer keyPosition;
		public List<DimensionValue> values;
	}

	private static class DimensionValue {
		public String id;
		public String name;
		// "start" and "end" are specific to the TIME_PERIOD dimension in SDMX-JSON
		public String start;
		public String end;
	}
}
